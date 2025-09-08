from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import time
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from module.model import ActorCritic
from module.buffer import EventBuffer, EventTransition
from module.smdp_gae import compute_smdp_gae


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
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # env / runtime
    seed: int = 7
    device: str = "cuda"

    # step-based training
    total_steps: int = 100_000
    steps_per_update: int = 4096

    # logging
    project: str = "smdp_mappo"


# -------------------- trainer --------------------
class Trainer:
    """
    Fixed-steps training:
      - collect_steps(min_steps): env.step을 고정 스텝만큼 돌려 샘플 수집
      - train_steps(total_steps, steps_per_update): 누적 스텝 기준 학습 진행
    ENV.step(action) -> (obs_next, reward, info_next)
      - obs: dict | None  # 데드락(의사결정) 구간에만 dict
      - info: {deadlock_active, event_start, event_end, tau, terminated, truncated, action_mask}
    """
    def __init__(self, cfg: TrainConfig, env):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 1) Env
        self.env = env
        obs, info = self.env.reset()

        # 2) Model 초기화 (초기 state_dim 확보)
        # deadlock이 없어서 obs=None일 수 있으므로 몇 스텝 굴려서 obs를 확보
        if obs is None:
            for _ in range(1000):
                obs, _, info = self.env.step(None)
                done = bool(info.get("terminated", False) or info.get("truncated", False))
                if done:
                    obs, info = self.env.reset()
                if obs is not None:
                    break
        assert isinstance(obs, dict) and "state" in obs, "ENV must return dict obs when deadlock active."
        state_dim = int(np.asarray(obs["state"]).shape[-1])

        self.model = ActorCritic(state_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

        # 3) W&B 준비
        run_base = (
            f"lr{_slug(cfg.lr)}_clip{_slug(cfg.clip_eps)}_ent{_slug(cfg.entropy_coef)}_"
            f"ep{cfg.epochs}_mb{cfg.minibatch_size}_steps{cfg.total_steps}_roll{cfg.steps_per_update}"
        )
        if wandb.run is None:
            wandb.init(project=cfg.project, config=vars(cfg), name=run_base)
        if wandb.run:
            wandb.run.name = run_base
        self._save_name = f"{run_base}.pt"

    # --------- 수집기: 고정 스텝 단위로 버퍼 채우기 ---------
    @torch.no_grad()
    def collect_steps(self, min_steps: int) -> Tuple[EventBuffer, int]:
        """
        최소 min_steps 만큼 env.step을 돌고,
        obs != None (의사결정 스텝)인 전이만 EventBuffer에 저장한다. (tau=1)
        """
        buf = EventBuffer(self.device)
        steps_used = 0

        obs, info = self.env.reset()
        while steps_used < min_steps:
            # 액션 (데드락 구간인 경우에만)
            if obs is not None:
                amask_np = info.get("action_mask", None)
                amask = None if amask_np is None else torch.as_tensor(
                    amask_np, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                action, logprob, value = self.model.act(obs, action_mask=amask)
                act_to_env = int(action.item())
            else:
                act_to_env = None

            # 전진
            obs_next, r, info_next = self.env.step(act_to_env)
            steps_used += 1

            # 전이 저장(데드락 구간에서만)
            if obs is not None:
                done = bool(info_next.get("terminated", False) or info_next.get("truncated", False))
                # 스텝 학습이므로 tau=1
                e = EventTransition(
                    state=torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device),
                    action=action.squeeze(0),
                    logprob=logprob.squeeze(0),
                    value=value.squeeze(0),
                    reward=float(r),
                    tau=1,
                    done=done,
                    terminated=bool(info_next.get("terminated", False)),
                    truncated=bool(info_next.get("truncated", False)),
                    action_mask=(amask.squeeze(0) if amask is not None else None),
                )
                idx = buf.add(e)

                # 부트스트랩 값 설정
                if (obs_next is not None) and (not done):
                    _, v_next = self.model.forward(obs_next)
                    buf.set_next(idx, next_state=None, next_value=v_next.squeeze(0))
                else:
                    # 데드락 해제/에피소드 끝 → V_next=0으로 컷
                    buf.set_next(idx, next_state=None, next_value=torch.zeros_like(value.squeeze(0)))

            # 경계 처리
            if bool(info_next.get("terminated", False) or info_next.get("truncated", False)):
                obs, info = self.env.reset()
            else:
                obs, info = obs_next, info_next

        return buf, steps_used

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
    def train(self, total_steps: Optional[int] = None, steps_per_update: Optional[int] = None):
        total_steps = total_steps or self.cfg.total_steps
        steps_per_update = steps_per_update or self.cfg.steps_per_update

        used = 0
        upd  = 0
        pbar = tqdm(total=total_steps, desc="steps", unit="step", ncols=0)

        while used < total_steps:
            upd += 1
            buf, steps_used = self.collect_steps(min_steps=min(steps_per_update, total_steps - used))
            used += steps_used
            pbar.update(steps_used)

            batch = buf.as_tensors()
            batch = compute_smdp_gae(batch, self.cfg.gamma, self.cfg.lam)
            logs  = self.update(batch)

            avgR   = float(batch["rewards"].mean().cpu()) if len(buf) else 0.0
            avgTau = float(batch["taus"].mean().cpu())    if len(buf) else 0.0

            # 상태바 우측에 최신 지표 표시
            pbar.set_postfix(
                upd=upd, avgR=f"{avgR:+.3f}", tau=f"{avgTau:.2f}",
            )

            # wandb 로깅 (누적 스텝을 step으로 사용)
            wandb.log({
                "train/avgR_event": avgR,
                "train/avgTau": avgTau,
                "train/loss": logs["loss"],
                "train/actor_loss": logs["actor_loss"],
                "train/value_loss": logs["value_loss"],
            }, step=used)

        pbar.close()

        # 저장
        os.makedirs("checkpoints", exist_ok=True)
        out_path = os.path.join("checkpoints", self._save_name)
        torch.save(self.model.state_dict(), out_path)
        print(f"[model] saved → {out_path}")
        wandb.finish()
