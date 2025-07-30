import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 공통 특징 추출기 (Common Feature Extractor)
class CommonNet(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# 2. 정책 헤드 (Policy Head)
class PolicyHead(nn.Module):
    def __init__(self, hidden_size=128, action_dims=5, action_levels=4):
        super().__init__()
        total_actor_outputs = action_dims * action_levels
        self.actor_head = nn.Linear(hidden_size, total_actor_outputs)
        self.action_dims = action_dims
        self.action_levels = action_levels

    def forward(self, x):
        action_logits_flat = self.actor_head(x)
        # 로짓을 [Batch, Dims, Levels] 형태로 재구성하여 그대로 반환합니다.
        action_logits = action_logits_flat.view(-1, self.action_dims, self.action_levels)
        return action_logits

# 3. 가치 헤드 (Value Head)
class ValueHead(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.critic_head(x)