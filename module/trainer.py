from dataclasses import dataclass
from typing import Optional, Dict, Any

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import ENV                     # <- 너의 ENV
from model import ActorCritic           # <- 너의 모델
from buffer import EventBuffer, EventTransition
from smdp_gae import compute_smdp_gae


@dataclass
class TrainConfig:
    prob_path: str
    total_updates: int = 1000
    events_per_update: int = 1024
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SMDPTrainer:
    """
    - ENV.step(action) -> (obs_next, reward, done, info_next)
      info_next keys(필수): event_start, event_end(또는 done 중 강제), tau(끝날 때),
                            terminated, truncated, action_mask
      obs/obs_next: dict {"state": np/torch, "edge_index": np/torch}
    """
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        # 1) Env
        self.env = ENV(cfg.prob_path)
        obs, info = self.env.reset()

        # 2) Model
        # obs가 None일 수 있으므로 하나 나올 때까지 굴림
        if obs is None:
            for _ in range(1000):
                obs, _, done, info = self.env.step(None)
                if done:
                    obs, info = self.env.reset()
                if obs is not None:
                    break
        assert isinstance(obs, dict) and "state" in obs, "ENV must return dict obs when deadlock active."
        state_dim = int(np.asarray(obs["state"]).shape[-1])
        self.model = ActorCritic(state_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

    # --------- 수집기: 이벤트 전이 단위로 버퍼 채우기 ---------
    @torch.no_grad()
    def collect_events(self, min_events: int) -> EventBuffer:
        buf = EventBuffer(self.device)
        pending: Optional[Dict[str, Any]] = None
        waiting_idx: Optional[int] = None

        obs, info = self.env.reset()
        done = False
        steps_guard = 0

        while len(buf) < min_events:
            steps_guard += 1
            if steps_guard > 1_000_000:
                raise RuntimeError("Collector guard exceeded. Check env for stalled run.")

            # ---- 의사결정/액션 (데드락 시점만) ----
            if obs is not None and info.get("event_start", False):
                # action mask
                amask_np = info.get("action_mask", None)
                amask = None
                if amask_np is not None:
                    amask = torch.as_tensor(amask_np, dtype=torch.bool, device=self.device)
                    if amask.dim() == 1:  # [A] -> [1,A]
                        amask = amask.unsqueeze(0)

                # 정책 호출 (dict 그대로)
                action, logprob, value = self.model.act(obs, action_mask=amask)

                pending = {
                    "state": torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device),
                    "action": action.squeeze(0),
                    "logprob": logprob.squeeze(0),
                    "value": value.squeeze(0),
                    "R": 0.0,
                    "mask": (amask.squeeze(0) if amask is not None else None),
                }
                act_to_env = int(action.item())
            else:
                act_to_env = None

            # ---- 한 스텝 전진 ----
            obs_next, r, done, info_next = self.env.step(act_to_env)

            # ---- 이벤트 진행 중이면 보상 누적 ----
            if pending is not None and info.get("deadlock_active", False):
                pending["R"] += float(r)

            # ---- 이벤트 종료/에피소드 종료: 전이 닫기 ----
            if pending is not None and (info_next.get("event_end", False) or done):
                tau_final = int(info_next["tau"])  # env가 계산한 τ 사용
                e = EventTransition(
                    state=pending["state"],
                    action=pending["action"],
                    logprob=pending["logprob"],
                    value=pending["value"],
                    reward=pending["R"],
                    tau=tau_final,
                    done=bool(done),
                    # ▼ 버퍼 dataclass에 다음 두 필드가 반드시 있어야 함
                    terminated=bool(info_next.get("terminated", False)),
                    truncated=bool(info_next.get("truncated", False)),
                    action_mask=pending["mask"],
                )
                idx = buf.add(e)
                waiting_idx = idx
                pending = None

            # ---- 다음 이벤트가 시작되면, 직전 전이에 next_value 채우기 ----
            if info_next.get("event_start", False) and waiting_idx is not None and obs_next is not None:
                # 부트스트랩을 위한 V(s_{k+1})
                _, v_next = self.model.forward(obs_next)   # 모델이 dict 입력을 지원
                buf.set_next(waiting_idx, next_state=None, next_value=v_next.squeeze(0))
                waiting_idx = None

            # ---- 에피소드 종료 처리 ----
            if done:
                obs, info = self.env.reset()
                done = False
            else:
                obs, info = obs_next, info_next

        return buf

    # --------- PPO 업데이트 ---------
    def update(self, batch: Dict[str, torch.Tensor]):
        cfg = self.cfg
        N = batch["rewards"].shape[0]
        idxs = torch.randperm(N, device=self.device)

        for _ in range(cfg.epochs):
            for start in range(0, N, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                # 미니배치 슬라이스
                states  = batch["states"][mb]
                actions = batch["actions"][mb]
                old_lp  = batch["old_logprobs"][mb]
                adv     = batch["advantages"][mb]
                ret     = batch["returns"][mb]
                amask   = None if batch["action_masks"] is None else batch["action_masks"][mb]

                # 새 logprob/value
                logp_new, entropy, value = self.model.crt({"state": states, "edge_index": None}, actions, amask)
                ratio = (logp_new - old_lp).exp()

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                actor_loss = -(torch.min(surr1, surr2)).mean()
                value_loss = 0.5 * (ret - value).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

    # --------- 전체 학습 루프 ---------
    def train(self):
        cfg = self.cfg
        for upd in range(1, cfg.total_updates + 1):
            t0 = time.time()
            buf = self.collect_events(cfg.events_per_update)
            batch = buf.as_tensors()
            batch = compute_smdp_gae(batch, cfg.gamma, cfg.lam)
            self.update(batch)
            dt = time.time() - t0

            avgR = float(batch["rewards"].mean().cpu())
            avgTau = float(batch["taus"].mean().cpu())
            print(f"[upd {upd:04d}] events={len(buf):5d}  avgR={avgR:+.3f}  avgTau={avgTau:.2f}  time={dt:.2f}s")
