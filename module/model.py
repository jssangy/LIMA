from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPEncoder(nn.Module):
    """
    단순 MLP 인코더.
    Inputs
      - x: [F] or [B, F]
    Output
      - [B, hidden]
    """
    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        blocks = []
        prev = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, F]
        return self.net(x)      # [B, hidden]


class ActorCritic(nn.Module):
    """
    입력 형태(ENV obs dict)를 그대로 유지:
      - inputs: {"state": [F] or [B,F], "edge_index": ... (무시됨)}
    """
    def __init__(self, state_dim: int, action_dim: int = 4, hidden: int = 128, mlp_layers: int = 2):
        super().__init__()
        self.encoder = MLPEncoder(state_dim, hidden, layers=mlp_layers)
        self.actor   = nn.Linear(hidden, action_dim)
        self.critic  = nn.Linear(hidden, 1)

    # ---- 입력 표준화: dict -> [B,F] 텐서 ----
    def _normalize_inputs(self, inputs: Dict[str, Any]) -> torch.Tensor:
        dev = self.actor.weight.device
        if isinstance(inputs, dict):
            x = torch.as_tensor(inputs["state"], dtype=torch.float32, device=dev)
            if x.ndim == 1:
                x = x.unsqueeze(0)  # [1,F]
            return x
        else:
            raise ValueError("Unsupported input type. Use dict with 'state' (edge_index is ignored).")

    # ---- 공통 forward ----
    def forward(self, inputs: Dict[str, Any]):
        x = self._normalize_inputs(inputs)
        z = self.encoder(x)
        logits = self.actor(z)                 # [B, A]
        value  = self.critic(z).squeeze(-1)    # [B]
        return logits, value

    @torch.no_grad()
    def act(self, inputs: Dict[str, Any], action_mask: Optional[torch.Tensor] = None):
        logits, value = self.forward(inputs)
        if action_mask is not None:
            if action_mask.dim() == 1 and logits.size(0) == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        dist = Categorical(logits=logits)
        action  = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def crt(self, inputs: Dict[str, Any], action: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        logits, value = self.forward(inputs)
        if action_mask is not None:
            if action_mask.dim() == 1 and logits.size(0) == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value
