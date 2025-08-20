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
from torch.distributions import Categorical

# ---------------------------------------------------------
# Plug your ENV here
# from your_module import ENV
# The ENV must implement the SMDP event flags as we discussed:
#   - step(action) -> obs_next (dict|None), reward(float), done(bool), info(dict)
#   - info contains: deadlock_active(bool), event_start(bool), event_end(bool), in_event(bool), tau(int when event_end True), terminated(bool), truncated(bool), action_mask(np.ndarray[bool]) when deadlock_active
#   - obs is a dict with at least {"state": np.ndarray(shape=[state_dim], dtype=float32)}
# For now we create a placeholder import path expecting ENV(prob_path)
try:
    from env import ENV  # adjust to your actual module
except Exception as e:
    ENV = None
    print("[WARN] Couldn't import ENV from env.py. Replace import path at top of this file.")

# ---------------------------------------------------------
# Utilities

def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device)

# ---------------------------------------------------------
# Actor-Critic (replace the encoder with your GNN later)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        # Simple MLP encoder; swap with your GNN body if needed
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        # state: [B, state_dim]
        z = self.encoder(state)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        logits, value = self.forward(state)
        if action_mask is not None:
            # mask: True = allowed, False = blocked
            # set -inf on invalid actions (use a large negative number to avoid NaNs)
            invalid = (~action_mask).to(torch.bool)
            logits = logits.masked_fill(invalid, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        logits, value = self.forward(state)
        if action_mask is not None:
            invalid = (~action_mask).to(torch.bool)
            logits = logits.masked_fill(invalid, -1e9)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value

# ---------------------------------------------------------
# Event Buffer for SMDP (stores per-event transitions)

@dataclass
class EventTransition:
    state: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    reward: float               # accumulated R_k
    tau: int                    # duration of the event
    done: bool
    terminated: bool
    truncated: bool
    action_mask: Optional[torch.Tensor] = None
    # next-state/value at next decision time (may be None if not yet observed)
    next_state: Optional[torch.Tensor] = None
    next_value: Optional[torch.Tensor] = None

class EventBuffer:
    def __init__(self, device: torch.device):
        self.device = device
        self.events: List[EventTransition] = []

    def add(self, e: EventTransition) -> int:
        self.events.append(e)
        return len(self.events) - 1

    def set_next(self, idx: int, next_state: torch.Tensor, next_value: torch.Tensor):
        self.events[idx].next_state = next_state
        self.events[idx].next_value = next_value

    def __len__(self):
        return len(self.events)

    def clear(self):
        self.events.clear()

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        # collate into tensors (pad missing next_value with 0)
        states = torch.stack([e.state for e in self.events]).to(self.device)
        actions = torch.stack([e.action for e in self.events]).to(self.device)
        old_logprobs = torch.stack([e.logprob for e in self.events]).to(self.device)
        values = torch.stack([e.value for e in self.events]).to(self.device)
        rewards = torch.tensor([e.reward for e in self.events], dtype=torch.float32, device=self.device)
        taus = torch.tensor([e.tau for e in self.events], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in self.events], dtype=torch.bool, device=self.device)
        terminated = torch.tensor([e.terminated for e in self.events], dtype=torch.bool, device=self.device)
        truncated = torch.tensor([e.truncated for e in self.events], dtype=torch.bool, device=self.device)

        # masks stored per-event (optional)
        if any(e.action_mask is not None for e in self.events):
            action_masks = torch.stack([e.action_mask if e.action_mask is not None else torch.ones_like(self.events[0].action_mask) for e in self.events]).to(self.device)
        else:
            action_masks = None

        next_values = []
        for e in self.events:
            if e.next_value is None:
                next_values.append(torch.tensor(0.0, device=self.device))
            else:
                next_values.append(e.next_value.to(self.device))
        next_values = torch.stack(next_values)

        return {
            "states": states,
            "actions": actions,
            "old_logprobs": old_logprobs,
            "values": values,
            "rewards": rewards,
            "taus": taus,
            "dones": dones,
            "terminated": terminated,
            "truncated": truncated,
            "action_masks": action_masks,
            "next_values": next_values,
        }

# ---------------------------------------------------------
# SMDP-GAE computation over event sequence

def compute_smdp_gae(batch: Dict[str, torch.Tensor], gamma: float, lam: float) -> Dict[str, torch.Tensor]:
    rewards = batch["rewards"]           # [N]
    taus = batch["taus"]                 # [N]
    values = batch["values"]             # [N]
    next_values = batch["next_values"]   # [N] (0 if unknown or terminal)
    dones = batch["dones"].float()       # [N]

    # delta_k = R_k + gamma^{tau_k} V(s_{k+1}) - V(s_k)
    gamma_tau = torch.pow(gamma, taus)
    deltas = rewards + gamma_tau * next_values - values

    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        lam_pow = torch.pow(gamma * lam, taus[t])
        last_gae = deltas[t] + lam_pow * last_gae * not_done
        advantages[t] = last_gae
    returns = advantages + values

    # normalize advantages (avoid NaN if var=0)
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    batch_out = batch.copy()
    batch_out["advantages"] = advantages
    batch_out["returns"] = returns
    return batch_out

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
