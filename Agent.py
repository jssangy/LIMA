import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, agent_id, state_size, action_size):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN 하이퍼파라미터
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        # DQN 네트워크
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        # D와 E 에이전트만 업데이트
        if self.agent_id not in ["D", "E"]:
            return
            
        # 메모리에 경험 저장
        self.memory.append((state, action, reward, next_state, done))
        
        # 메모리가 충분히 쌓이면 학습 시작
        if len(self.memory) >= self.batch_size:
            self.train()
            
        # 에피소드가 끝나면 타겟 네트워크 업데이트
        if done:
            self.update_target_network()
            
        # epsilon 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self):
        # 배치 샘플링
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 텐서 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 현재 Q 값
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 다음 상태의 최대 Q 값
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 손실 계산 및 역전파
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 