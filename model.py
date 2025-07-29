import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, repulsive_dims=4, force_levels=4, center_dims=4):
        super(ActorCritic, self).__init__()
        self.repulsive_dims = repulsive_dims
        self.force_levels = force_levels
        self.center_dims = center_dims

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor Head: (4*4 for forces) + (4 for center direction) = 20 logits
        total_actor_outputs = (self.repulsive_dims * self.force_levels) + self.center_dims
        self.actor_head = nn.Linear(128, total_actor_outputs)
        
        # Critic Head: 상태 가치를 출력
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        
        # 로짓을 반발장 부분과 중심 제어 부분으로 분리
        repulsive_logits_flat = action_logits[:, :self.repulsive_dims * self.force_levels]
        center_logits = action_logits[:, self.repulsive_dims * self.force_levels:]
        
        # 반발장 로짓을 [batch_size, 4, 4] 형태로 재구성
        repulsive_logits = repulsive_logits_flat.view(-1, self.repulsive_dims, self.force_levels)
        
        return (repulsive_logits, center_logits), state_value