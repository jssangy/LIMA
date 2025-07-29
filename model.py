import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims=4, num_force_levels=4):
        super(ActorCritic, self).__init__()
        self.action_dims = action_dims # 방향의 수 (N, E, S, W) = 4
        self.num_force_levels = num_force_levels # 반발장 세기 레벨 (0, 1, 2, 3) = 4

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor Head: 4방향 * 4레벨 = 16개의 로짓을 출력
        self.actor_head = nn.Linear(128, self.action_dims * self.num_force_levels)
        
        # Critic Head: 상태 가치를 출력
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor: [batch_size, 16] 형태의 로짓 출력
        action_logits = self.actor_head(x)
        
        # Critic: [batch_size, 1] 형태의 상태 가치 출력
        state_value = self.critic_head(x)
        
        # 로짓을 [batch_size, 4, 4] 형태로 재구성하여 반환
        return action_logits.view(-1, self.action_dims, self.num_force_levels), state_value