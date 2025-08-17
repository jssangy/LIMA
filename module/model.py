import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNBody(nn.Module):
    """
    공유 GNN 몸체 (Shared GNN Body)
    - 입력: (배치된) 서브그래프의 노드 특징(x)과 엣지 인덱스(edge_index)
    - 출력: (배치된) 서브그래프의 각 노드에 대한 상황 인식 임베딩
    """
    def __init__(self, input_dim=24, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # GNN 레이어를 통해 정보 전파 (Message Passing)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Actor(nn.Module):
    """
    분산 액터 헤드 (Decentralized Actor Head)
    - 입력: 단일 노드의 임베딩 벡터
    - 출력: 해당 노드의 행동 확률 분포 (Logits)
    """
    def __init__(self, hidden_dim=128, action_dims=5, action_levels=4):
        super().__init__()
        self.action_dims = action_dims
        self.action_levels = action_levels
        total_outputs = action_dims * action_levels

        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, total_outputs)
        )

    def forward(self, node_embeddings):
        # GNN으로부터 받은 노드별 임베딩에 대해 각각 행동 로짓 계산
        action_logits_flat = self.layer(node_embeddings)
        # 로짓을 [Total_Nodes_in_Batch, Dims, Levels] 형태로 재구성
        action_logits = action_logits_flat.view(-1, self.action_dims, self.action_levels)
        return action_logits

class Critic(nn.Module):
    """
    torch_geometric 배치를 처리하는 중앙화된 크리틱 헤드
    - 입력: 배치 내 모든 노드의 임베딩 벡터와 배치 정보 벡터
    - 출력: 각 서브그래프의 상태 가치 (배치 크기만큼의 값)
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, all_node_embeddings, batch):
        # [수정] global_mean_pool을 사용하여 각 그래프의 평균 임베딩을 계산
        # all_node_embeddings: [Total_Nodes_in_Batch, Hidden_Dim]
        # batch: [Total_Nodes_in_Batch]
        # 출력: [Batch_Size, Hidden_Dim]
        global_state_embedding = global_mean_pool(all_node_embeddings, batch)
        
        # 전역 임베딩을 사용하여 상태 가치 평가
        return self.layer(global_state_embedding)

class GNN_PPO_Agent(nn.Module):
    """
    torch_geometric 배치를 처리하는 GNN 기반 PPO 에이전트
    """
    def __init__(self, input_dim=24, hidden_dim=128, action_dims=5, action_levels=4):
        super().__init__()
        self.gnn_body = GNNBody(input_dim, hidden_dim)
        self.actor = Actor(hidden_dim, action_dims, action_levels)
        self.critic = Critic(hidden_dim)

    def forward(self, nodes, edge_index, batch):
        """
        torch_geometric의 batch 벡터를 추가로 입력받음
        """
        # 1. GNN 몸체를 통해 노드 임베딩 생성
        node_embeddings = self.gnn_body(nodes, edge_index)

        # 2. 액터 헤드를 통해 행동 로짓 계산
        action_logits = self.actor(node_embeddings)

        # 3. 크리틱 헤드를 통해 상태 가치 계산
        state_value = self.critic(node_embeddings, batch)

        return action_logits, state_value
