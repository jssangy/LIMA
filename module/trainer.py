from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import types
import os
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from module.model import ActorCritic
from module.buffer import EventBuffer, EventTransition
from module.gae import compute_gae
from Environment import ENV


# -------------------- utils --------------------
def _slug(v):
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    return s.replace(".", "p").replace("-", "m")


# -------------------- config --------------------
@dataclass
class TrainConfig:
    # PPO / optimization
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 1e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # env / runtime
    device: str = "cuda"

    # step-based training
    total_steps: int = 500_000
    events_per_update: int = 256

    # curriculum
    curriculum_min_tasks: int = 3
    curriculum_max_tasks: int = 15
    solves_per_stage: int = 100          # 각 N에서 100회 완전 해결하면 다음 스테이지로

    # logging
    project: str = "MAPF"

    # env parameters (for creating ENV inside Trainer)
    prob_path: str = "problems/cross/cross_1.json"
    max_arm_h: int = 5
    max_arm_v: int = 5
    num_amrs: int = 3
    max_steps: int = 1024
    running_opt: int = 0  # 0=BFS, 1=A*, 2=D*, 3=PIBT, 4=CBS
    traffic_mode: str = "task"  # "traffic" or "task"


# -------------------- trainer --------------------
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 1) Env
        self.env = ENV(cfg.prob_path, cfg.max_arm_h, cfg.max_arm_v, cfg.num_amrs, cfg.max_steps, cfg.running_opt, cfg.traffic_mode)
        self.obs_state, self.info_state = self.env.reset()

        self.iid = next(iter(self.obs_state.keys()))

        # 2) Model 초기화 (초기 state_dim 확보)
        obs_i = self.obs_state.get(self.iid)
        if obs_i is None:
            for _ in range(1000):
                obs, _, info = self.env.step(None)
                summary = info.get("_summary", {})
                if bool(summary.get("terminated", False) or summary.get("truncated", False)):
                    obs, info = self.env.reset()
                obs_i = obs.get(self.iid)
                if obs_i is not None:
                    break

        assert isinstance(obs_i, dict) and "state" in obs_i, "ENV must return dict obs when deadlock active."
        state_dim = int(np.asarray(obs_i["state"]).shape[-1])

        self.model = ActorCritic(state_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

        # 3) W&B 준비
        run_base = (
            f"lr{_slug(cfg.lr)}_clip{_slug(cfg.clip_eps)}_ent{_slug(cfg.entropy_coef)}_"
            f"ep{cfg.epochs}_mb{cfg.minibatch_size}_steps{cfg.total_steps}_roll{cfg.events_per_update}"  # 여기서 steps_per_update를 events_per_update로 변경
        )
        if wandb.run is None:
            wandb.init(project=cfg.project, config=vars(cfg), name=run_base)
        if wandb.run:
            wandb.run.name = run_base
        self._save_name = f"{run_base}.pt"

        self.solved_count = 0

    # --------- 수집기: 이벤트 단위로 버퍼 채우기 ---------
    @torch.no_grad()
    def collect_macro_events(self, min_events: int) -> Tuple[EventBuffer, int]:
        buf = EventBuffer(self.device)
        events_collected = 0

        obs, info = self.obs_state, self.info_state
        episodes_seen = 0  # 이번 수집 싸이클 동안 지나간 에피소드 수

        while events_collected < min_events:
            obs_i  = obs.get(self.iid)
            info_i = info.get(self.iid)

            # 매크로 동작 중이면 tick만
            if info_i.get("macro_busy", False):
                obs, _, info = self.env.step(None)

            # 데드락이면 의사결정 + 이벤트 기록
            elif info_i.get("is_deadlock", False):
                amask = torch.as_tensor(info_i["action_mask"], dtype=torch.bool, device=self.device)
                action, logprob, value = self.model.act(obs_i, action_mask=amask)
                obs, reward_map, info = self.env.step({self.iid: int(action.item())})
                reward_i = float(reward_map.get(self.iid, 0.0))

                summary = info.get("_summary", {})
                done = bool(summary.get("terminated", False) or summary.get("truncated", False))

                e = EventTransition(
                    state=torch.as_tensor(obs_i["state"], dtype=torch.float32, device=self.device),
                    action=action.squeeze(0),
                    logprob=logprob.squeeze(0),
                    value=value.squeeze(0),
                    reward=reward_i,
                    tau=1,
                    done=done,
                    terminated=bool(summary.get("terminated", False)),
                    truncated=bool(summary.get("truncated", False)),
                    action_mask=amask
                )
                buf.add(e)
                events_collected += 1

            # 그 외에는 환경만 진행
            else:
                obs, _, info = self.env.step(None)

            # --- 에피소드 경계 처리 (여기가 핵심 수정) ---
            summary = info.get("_summary", {})
            done = bool(summary.get("terminated", False) or summary.get("truncated", False))
            if done:
                # 성공 판정: spawned_total == completed_total (>0)
                solved = False
                if getattr(self.env, "traffic_mode", "") == "task" and hasattr(self.env, "task_generator"):
                    prog = self.env.task_generator.get_progress()
                    spawned   = int(prog.get("spawned_total", 0))
                    completed = int(prog.get("completed_total", 0))
                    solved = (spawned > 0 and spawned == completed)
                if solved:
                    self.solved_count += 1

                obs, info = self.env.reset()
                episodes_seen += 1

                # 이번 수집 싸이클에서 이벤트가 하나도 없었다면 즉시 반환하여 바깥 pbar가 진행되게 함
                if events_collected == 0:
                    break

        self.obs_state, self.info_state = obs, info
        return buf, events_collected


    
    # --------- PPO 업데이트 ---------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        N = batch["rewards"].shape[0]
        if N == 0:
            # 학습할 전이가 없는 경우(의사결정 스텝이 한 번도 없었음)
            return {"loss": 0.0, "actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "entropy_loss": 0.0}

        idxs = torch.randperm(N, device=self.device)

        tot_loss = tot_actor = tot_value = tot_entropy = 0.0
        n_mb = 0

        for _ in range(cfg.epochs):
            for start in range(0, N, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                states  = batch["states"][mb]
                actions = batch["actions"][mb]
                old_lp  = batch["old_logprobs"][mb]
                adv     = batch["advantages"][mb]
                ret     = batch["returns"][mb]
                amask   = None if batch["action_masks"] is None else batch["action_masks"][mb]

                logp_new, entropy, value = self.model.crt({"state": states, "edge_index": None}, actions, amask)
                ratio = (logp_new - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv

                actor_loss    = -(torch.min(surr1, surr2)).mean()
                value_loss    = 0.5 * (ret - value).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                # 누적
                n_mb        += 1
                tot_loss    += loss.item()
                tot_actor   += actor_loss.item()
                tot_value   += value_loss.item()
                tot_entropy += entropy_bonus.item()

        # 평균값들
        avg_loss    = tot_loss    / n_mb
        avg_actor   = tot_actor   / n_mb
        avg_value   = tot_value   / n_mb
        avg_entropy = tot_entropy / n_mb
        avg_entropy_loss = - cfg.entropy_coef * avg_entropy  # 최종 로스에 기여하는 항(음수)

        return {
            "loss": avg_loss,
            "actor_loss": avg_actor,
            "value_loss": avg_value,
            "entropy": avg_entropy,
            "entropy_loss": avg_entropy_loss,
        }

    def train(self):
        events_per_update = self.cfg.events_per_update
        os.makedirs("checkpoint", exist_ok=True)
        best = {"avgR": -float("inf"), "upd": -1, "stage": -1}
        best_path = os.path.join("checkpoint", "best_policy.pt")
        global_upd = 0

        for tasks_per_ep in range(self.cfg.curriculum_min_tasks, self.cfg.curriculum_max_tasks + 1):
            self.env = ENV(
                self.cfg.prob_path,
                self.cfg.max_arm_h,
                self.cfg.max_arm_v,
                tasks_per_ep,
                self.cfg.max_steps,
                self.cfg.running_opt,
                self.cfg.traffic_mode
            )
            self.obs_state, self.info_state = self.env.reset()
            self.iid = next(iter(self.obs_state.keys()))

            stage_target = 1000 if tasks_per_ep <= 8 else 100   # 16 이하일 땐 1000회, 그 이상은 100회

            self.solved_count = 0
            prev_solved = 0
            pbar = tqdm(
                total=stage_target,                              # ➋ pbar 총합 변경
                desc=f"[{tasks_per_ep} tasks] solved",
                unit="episode",
                ncols=0
            )

            # 이 스테이지를 stage_target 회 해결할 때까지 반복
            while self.solved_count < stage_target:             # ➌ 종료 조건 변경
                buf, events_used = self.collect_macro_events(min_events=events_per_update)

                if events_used == 0:
                    if self.solved_count > prev_solved:
                        pbar.update(self.solved_count - prev_solved)
                        prev_solved = self.solved_count
                    pbar.set_postfix(
                        upd=global_upd,
                        loss="—",
                        avgR="—",
                        solved=f"{self.solved_count}/{stage_target}"   # 표기도 stage_target 사용
                    )
                    continue

                # PPO 업데이트 ...
                batch = buf.as_tensors()
                batch = compute_gae(batch, self.cfg.gamma, self.cfg.lam)
                logs  = self.update(batch)
                global_upd += 1

                avgR     = float(batch["rewards"].mean().cpu()) if len(buf) else 0.0
                cur_loss = float(logs.get("loss", float("inf")))

                if avgR > best["avgR"] + 1e-8:
                    best.update(avgR=avgR, upd=global_upd, stage=tasks_per_ep)
                    torch.save(self.model.state_dict(), best_path)
                    if wandb.run:
                        wandb.summary["best/avgR"]   = avgR
                        wandb.summary["best/update"] = global_upd
                        wandb.summary["best/stage"]  = tasks_per_ep

                if self.solved_count > prev_solved:
                    pbar.update(self.solved_count - prev_solved)
                    prev_solved = self.solved_count

                pbar.set_postfix(
                    upd=global_upd,
                    loss=f"{cur_loss:.4f}",
                    avgR=f"{avgR:+.3f}",
                    solved=f"{self.solved_count}/{stage_target}"
                )

                wandb.log({
                    "train/avgR_event": avgR,
                    "train/loss": logs["loss"],
                    "train/actor_loss": logs["actor_loss"],
                    "train/value_loss": logs["value_loss"],
                    "curriculum/tasks_per_ep": tasks_per_ep,
                    "curriculum/solved_in_stage": self.solved_count,
                    "curriculum/stage_target": stage_target,      # 로깅도 같이
                }, step=global_upd)

            pbar.close()

        final_path = os.path.join("checkpoint", "final_policy.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"[Final model] saved → {final_path}")
        print(f"[Best model]  saved → {best_path} (avgR={best['avgR']:.6f}, stage={best['stage']}, upd={best['upd']})")
        wandb.finish()