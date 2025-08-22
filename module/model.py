from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphEncoder(nn.Module):
    """
    GCN + global mean pooling.
    Inputs
      - x: [N, in_dim]
      - edge_index: [2, E] (long)  — 빈/None이면 self-loop로 대체
      - batch: [N]  — None이면 zeros(N)로 대체(그래프 1개)
    Output
      - [B, hidden]
    """
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(layers - 1)]
        )
        self.act = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = x.device
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # 엣지 없으면 self-loop 주입(단일 노드 포함 모든 케이스 GCN 경로 보장)
        if edge_index is None or edge_index.numel() == 0:
            idx = torch.arange(x.size(0), dtype=torch.long, device=device)
            edge_index = torch.stack([idx, idx], dim=0)
        else:
            edge_index = edge_index.to(device=device, dtype=torch.long)

        h = x
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        return global_mean_pool(h, batch)  # [B, hidden]


class ActorCritic(nn.Module):
    """
    입력 형태를 유연하게 받는 정책-가치 네트워크.
    - dict: {"state": [F] or [N,F], "edge_index": [2,E] (선택)}
    - tuple: (state, edge_index)
    - tensor: state만 [F] 또는 [B,F]  → B개의 1-노드 그래프로 자동 변환
    """
    def __init__(self, state_dim: int, action_dim: int = 4, hidden: int = 128, gnn_layers: int = 2):
        super().__init__()
        self.encoder = GraphEncoder(state_dim, hidden, layers=gnn_layers)
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    # ---- 입력 표준화: 어떤 형태든 (x, edge_index, batch)로 변환 ----
    def _normalize_inputs(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.actor.weight.device

        # dict (ENV obs 형태)
        if isinstance(inputs, dict):
            x = torch.as_tensor(inputs["state"], dtype=torch.float32, device=dev)
            if x.ndim == 1:
                x = x.unsqueeze(0)  # [1,F]
            ei_in = inputs.get("edge_index", None)
            if ei_in is None:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
            else:
                edge_index = torch.as_tensor(ei_in, dtype=torch.long, device=dev)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=dev)
            return x, edge_index, batch
        else:
            raise ValueError("Unsupported input type. Use dict with 'state' and optional 'edge_index'.")

    # ---- 공통 forward ----
    def forward(self, inputs: Dict[str, Any]):
        x, edge_index, batch = self._normalize_inputs(inputs)
        z = self.encoder(x, edge_index, batch)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, inputs: Dict[str, Any], action_mask: Optional[torch.Tensor] = None):
        logits, value = self.forward(inputs)
        if action_mask is not None:
            # action_mask: True=가능 / False=불가, shape [B, A] 또는 [A]
            if action_mask.dim() == 1 and logits.size(0) == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
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
