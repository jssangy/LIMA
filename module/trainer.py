from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import time
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
    minibatch_size: int = 128
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
    events_per_update: int = 128

    # logging
    project: str = "MAPF"


# -------------------- trainer --------------------
class Trainer:
    def __init__(self, cfg: TrainConfig, env):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 1) Env
        self.env = env
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

        self.event_on = False
        self.tau_event = 0

    # --------- 수집기: 이벤트 단위로 버퍼 채우기 ---------
    @torch.no_grad()
    def collect_macro_events(self, min_events: int) -> Tuple[EventBuffer, int]:  # min_steps -> min_events로 변경
        """
        - 환경에서 macro가 종료될 때까지 기다린 후 한 번만 데이터를 수집
        - macro가 완료될 때까지 action을 반복적으로 실행하고, 종료된 시점에 reward를 저장
        """
        buf = EventBuffer(self.device)
        events_collected = 0
        steps = 0

        # [수정] 초기 상태 가져오기
        obs, info = self.obs_state, self.info_state

        while events_collected < min_events:  # min_steps -> min_events로 변경
            steps += 1
            obs_i = obs.get(self.iid)
            info_i = info.get(self.iid)

            # macro가 실행 중일 경우, macro가 끝날 때까지 기다림
            if info_i["macro_busy"]:
                obs_next, _, info_next = self.env.step(None)
                obs, info = obs_next, info_next
                continue

            # deadlock이 발생했을 때
            if info_i["is_deadlock"]:
                amask = torch.as_tensor(info_i["action_mask"], dtype=torch.bool, device=self.device)
                action, logprob, value = self.model.act(obs_i, action_mask=amask)
                act_to_env = {self.iid: int(action.item())}

                # action을 실행하고, macro가 끝날 때까지 기다림
                obs_next, reward_map, info_next = self.env.step(act_to_env)
                reward_i = float(reward_map.get(self.iid, 0.0))

                summary = info_next.get("_summary", {})
                done = bool(summary.get("terminated", False) or summary.get("truncated", False))

                # Event 생성
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
                obs, info = obs_next, info_next

            # deadlock이 아닐 경우, 다음 상태로 이동
            else:
                obs_next, _, info_next = self.env.step(None)
                obs, info = obs_next, info_next

            # 환경이 종료되었으면 리셋
            done = bool(info.get("terminated", False) or info.get("truncated", False))
            if done:
                obs, info = self.env.reset()

        self.obs_state, self.info_state = obs, info
        return buf, events_collected, steps

    
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

    # --------- 전체 학습 루프 (고정 스텝) ---------
    def train(self, total_steps: Optional[int] = None, events_per_update: Optional[int] = None):
        total_steps = total_steps or self.cfg.total_steps
        events_per_update = events_per_update or self.cfg.events_per_update

        os.makedirs("checkpoint", exist_ok=True)
        best = {"avgR": -float("inf"), "step": -1, "upd": -1}
        best_path = os.path.join("checkpoint", "best_policy.pt")

        used = 0
        upd  = 0
        
        # 진행 상황을 추적할 수 있는 pbar 추가
        pbar = tqdm(total=total_steps, desc="Training Progress", unit="step", ncols=0)

        while used < total_steps:
            upd += 1

            # 이벤트 단위로 수집하는 collect_macro_events 호출
            buf, events_used, steps_used = self.collect_macro_events(min_events=events_per_update)
            used += steps_used
            pbar.update(events_used)  # 진행상황 업데이트

            if events_used > 0:
                batch = buf.as_tensors()
                batch = compute_gae(batch, self.cfg.gamma, self.cfg.lam)
                logs  = self.update(batch)

                # 베스트(최소 loss) 저장
                cur_loss = float(logs.get("loss", float("inf")))
                avgR   = float(batch["rewards"].mean().cpu()) if len(buf) else 0.0

                # [수정] 베스트(최대 avgR) 저장
                if avgR > best["avgR"] + 1e-8:
                    best.update(avgR=avgR, step=used, upd=upd)
                    torch.save(self.model.state_dict(), best_path)
                    if wandb.run:
                        wandb.summary["best/avgR"] = avgR
                        wandb.summary["best/step"] = used
                        wandb.summary["best/update"] = upd

                # 상태바 우측에 최신 지표 표시
                pbar.set_postfix(
                    upd=upd, loss=f"{cur_loss:.4f}", avgR=f"{avgR:+.3f}"
                )

                # wandb 로깅 (누적 스텝을 step으로 사용)
                wandb.log({
                    "train/avgR_event": avgR,
                    "train/loss": logs["loss"],
                    "train/actor_loss": logs["actor_loss"],
                    "train/value_loss": logs["value_loss"],
                }, step=used)

        pbar.close()

        # 저장
        os.makedirs("checkpoint", exist_ok=True)
        out_path = os.path.join("checkpoint", 'final_policy.pt')
        torch.save(self.model.state_dict(), out_path)
        print(f"[Final model] saved → {out_path}")
        print(f"[Best model] saved → {best_path} (avgR={best['avgR']:.6f}, step={best['step']}, upd={best['upd']})")
        wandb.finish()
