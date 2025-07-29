import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    PPO를 위한 Actor-Critic 신경망 모델.
    상태를 입력받아 행동 확률 분포와 상태 가치를 출력합니다.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        신경망 레이어를 초기화합니다.

        :param state_dim: 상태 벡터의 차원 (29)
        :param action_dim: 행동의 개수 (예: 2 또는 4)
        :param hidden_dim: 은닉층의 뉴런 수
        """
        super(ActorCritic, self).__init__()

        # Actor와 Critic이 공유하는 공통 레이어
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor Head: 정책(어떤 행동을 할지)을 결정
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic Head: 상태의 가치를 평가
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        순전파를 수행합니다.

        :param state: (배치 크기, state_dim) 모양의 텐서
        :return: 행동 분포 (Categorical), 상태 가치 (Tensor)
        """
        # 공통 레이어를 통과시켜 특징 추출
        shared_features = self.shared_layers(state)

        # Actor Head를 통과시켜 행동 확률 계산
        action_logits = self.actor_head(shared_features)
        action_dist = Categorical(logits=action_logits)

        # Critic Head를 통과시켜 상태 가치 계산
        state_value = self.critic_head(shared_features)

        return action_dist, state_value

    def get_action(self, state):
        """
        주어진 상태에 대해 행동, 로그 확률, 상태 가치를 반환합니다.
        (실제 행동 선택 시 사용)
        """
        action_dist, state_value = self.forward(state)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, state_value

    def evaluate_action(self, state, action):
        """
        주어진 상태와 행동에 대해 로그 확률, 상태 가치, 엔트로피를 반환합니다.
        (학습 시 사용)
        """
        action_dist, state_value = self.forward(state)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return log_prob, state_value, entropy