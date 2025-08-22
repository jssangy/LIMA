from dataclasses import dataclass
from typing import Optional, List, Dict
import torch

@dataclass
class EventTransition:
    state: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    reward: float               # accumulated R_k
    tau: int                    # duration of the event
    done: bool
    terminated: bool = False
    truncated: bool = False
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