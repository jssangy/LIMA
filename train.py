import os
import math
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Environment import ENV

# ---------------------------------------------------------
# Plug your ENV here
# from your_module import ENV
# The ENV must implement the SMDP event flags as we discussed:
#   - step(action) -> obs_next (dict|None), reward(float), done(bool), info(dict)
#   - info contains: deadlock_active(bool), event_start(bool), event_end(bool), in_event(bool), tau(int when event_end True), terminated(bool), truncated(bool), action_mask(np.ndarray[bool]) when deadlock_active
#   - obs is a dict with at least {"state": np.ndarray(shape=[state_dim], dtype=float32)}
# For now we create a placeholder import path expecting ENV(prob_path)

# ---------------------------------------------------------
# Utilities

def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device)

# ---------------------------------------------------------
# Training loop (collector builds per-event transitions)

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
    weight_tau: bool = True          # weight samples by sqrt(tau)

class SMDPTrainer:
    def __init__(self, cfg: TrainConfig):
        assert ENV is not None, "Replace the ENV import path to your implementation."
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Init env
        self.env = ENV(cfg.prob_path)
        # Reset to get state dim
        obs, info = self.env.reset()
        if obs is None:
            # If no deadlock at reset, step with None until we get an obs or reach a safeguard
            for _ in range(100):
                obs, _, done, info = self.env.step(None)
                if done:
                    obs, info = self.env.reset()
                if obs is not None:
                    break
        assert obs is None or isinstance(obs, dict), "ENV must return dict obs or None"
        state_dim = int(obs["state"].shape[-1]) if obs is not None else 24  # fallback
        action_dim = 4  # N,E,S,W

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def collect_events(self, min_events: int) -> EventBuffer:
        buf = EventBuffer(self.device)
        pending: Optional[Dict[str, Any]] = None  # current open event
        waiting_idx: Optional[int] = None         # last event index waiting for next_state/value

        # Get initial observation
        obs, info = self.env.reset()
        done = False
        steps_guard = 0

        while len(buf) < min_events:
            steps_guard += 1
            if steps_guard > 1_000_000:
                raise RuntimeError("Collector ran too long without enough events. Check env or config.")

            # Decide action for CURRENT observation (event states only)
            if obs is not None and info.get("deadlock_active", False):
                state_t = to_tensor(obs["state"], self.device).float().unsqueeze(0)
                mask_np = info.get("action_mask", None)
                mask_t = None if mask_np is None else to_tensor(mask_np, self.device).bool().unsqueeze(0)

                action_t, logprob_t, value_t = self.model.act(state_t, mask_t)
                action_np = action_t.squeeze(0).cpu().numpy().item()
                # If a new event starts at this decision time, open a pending
                if info.get("event_start", False):
                    pending = {
                        "state": state_t.squeeze(0),
                        "action": action_t.squeeze(0),
                        "logprob": logprob_t.squeeze(0),
                        "value": value_t.squeeze(0),
                        "action_mask": (mask_t.squeeze(0) if mask_t is not None else None),
                        "R": 0.0,
                        "tau": 0,
                    }
                # Apply action this step
                act_to_env = int(action_np)
            else:
                act_to_env = None

            # Step env
            obs_next, reward, done, info_next = self.env.step(act_to_env)

            # Accumulate reward into pending event if any
            if pending is not None and info.get("deadlock_active", False):
                pending["R"] += float(reward)
                pending["tau"] += 1

            # If event ended OR episode ended, close the event (without next state yet)
            closed_now = False
            if pending is not None and (info_next.get("event_end", False) or done):
                e = EventTransition(
                    state=pending["state"],
                    action=pending["action"],
                    logprob=pending["logprob"],
                    value=pending["value"],
                    reward=float(pending["R"]),
                    tau=int(pending["tau"]),
                    done=bool(done),
                    terminated=bool(info_next.get("terminated", False)),
                    truncated=bool(info_next.get("truncated", False)),
                    action_mask=pending.get("action_mask", None),
                )
                idx = buf.add(e)
                waiting_idx = idx
                pending = None
                closed_now = True

            # If a new event starts now (next obs is decision state), we can fill next_state/value of the previous event
            if info_next.get("event_start", False) and waiting_idx is not None and obs_next is not None:
                ns = to_tensor(obs_next["state"], self.device).float().unsqueeze(0)
                with torch.no_grad():
                    _, nv = self.model.forward(ns)
                buf.set_next(waiting_idx, ns.squeeze(0), nv.squeeze(0))
                waiting_idx = None

            # If episode finished, start a new one
            if done:
                obs, info = self.env.reset()
                done = False
                continue

            # Roll to next step
            obs, info = obs_next, info_next

        return buf

    def update(self, batch: Dict[str, torch.Tensor]):
        cfg = self.cfg
        # Optional sample weighting by sqrt(tau) to reflect longer events a bit more
        if cfg.weight_tau:
            weights = torch.sqrt(batch["taus"].clamp(min=1.0))
        else:
            weights = torch.ones_like(batch["rewards"]) 

        N = batch["states"].shape[0]
        idxs = torch.randperm(N, device=self.device)
        for _ in range(cfg.epochs):
            for start in range(0, N, cfg.minibatch_size):
                mb_idx = idxs[start:start+cfg.minibatch_size]
                s = batch["states"][mb_idx]
                a = batch["actions"][mb_idx]
                adv = batch["advantages"][mb_idx]
                ret = batch["returns"][mb_idx]
                old_lp = batch["old_logprobs"][mb_idx]
                mb_w = weights[mb_idx]
                amask = None if batch["action_masks"] is None else batch["action_masks"][mb_idx]

                new_lp, entropy, value = self.model.evaluate_actions(s, a, amask)
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                actor_loss = -torch.mean(torch.min(surr1, surr2) * mb_w)

                value_loss = 0.5 * ((ret - value) ** 2 * mb_w).mean()
                entropy_bonus = entropy.mean()

                loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

    def train(self):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        for update in range(1, cfg.total_updates + 1):
            t0 = time.time()
            buf = self.collect_events(cfg.events_per_update)
            batch = buf.as_tensors()
            batch = compute_smdp_gae(batch, cfg.gamma, cfg.lam)
            self.update(batch)
            dt = time.time() - t0

            # Simple logging
            avg_R = float(batch["rewards"].mean().cpu())
            avg_tau = float(batch["taus"].mean().cpu())
            print(f"[upd {update:04d}] events={len(buf):5d}  avgR={avg_R:+.3f}  avgTau={avg_tau:.2f}  time={dt:.2f}s")

# ---------------------------------------------------------
# Entry

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_path", type=str, required=True, help="Path to your problem JSON for ENV(prob_path)")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--events_per_update", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainConfig(
        prob_path=args.prob_path,
        total_updates=args.updates,
        events_per_update=args.events_per_update,
        epochs=args.epochs,
        minibatch_size=args.minibatch,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip,
        lr=args.lr,
        device=args.device,
    )
    trainer = SMDPTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
