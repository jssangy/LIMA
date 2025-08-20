import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# GNNBody 클래스는 기존과 동일합니다.
class GNNBody(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    @torch.no_grad()
    def _compact(self, x, edge_index, active_mask):
        active_mask = active_mask.bool()
        keep_idx = torch.nonzero(active_mask, as_tuple=True)[0]      # [N_active]
        if keep_idx.numel() == 0:
            x_sub = x.new_zeros((0, x.size(-1)))
            ei_sub = edge_index.new_zeros((2, 0))
            return x_sub, ei_sub, keep_idx
        new_id = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        new_id[keep_idx] = torch.arange(keep_idx.numel(), device=x.device)
        ei = edge_index
        m = (new_id[ei[0]] >= 0) & (new_id[ei[1]] >= 0)              # 활성 엣지만
        ei_sub = torch.stack((new_id[ei[0, m]], new_id[ei[1, m]]), 0)
        return x[keep_idx], ei_sub, keep_idx

    def forward(self, x, edge_index, active_nodes):
        x_sub, ei_sub, keep_idx = self._compact(x, edge_index, active_nodes)
        if x_sub.size(0) == 0:
            return x.new_zeros((0, self.hidden_dim)), keep_idx
        h = F.relu(self.conv1(x_sub, ei_sub))
        h = F.relu(self.conv2(h, ei_sub))
        return h, keep_idx


class Actor(nn.Module):
    def __init__(self, hidden_dim=128, action_dims=4):
        super().__init__()
        self.action_dims = action_dims
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dims)
        )

    def forward(self, node_embeds, batch_vec=None):
        if node_embeds.size(0) == 0:
            return node_embeds.new_zeros((1, self.action_dims))  # 빈 그래프 안전 처리
        if batch_vec is None:                                    # 단일 서브그래프
            g = node_embeds.mean(dim=0, keepdim=True)
        else:                                                    # 여러 서브그래프
            g = global_mean_pool(node_embeds, batch_vec)
        return self.net(g)                                       # [B, action_dims]

# Critic 클래스는 이미 이 구조를 따르고 있으므로 기존과 동일합니다.
class Critic(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, all_node_embeddings, batch_vector):
        global_state_embedding = global_mean_pool(all_node_embeddings, batch_vector)
        return self.layer(global_state_embedding)