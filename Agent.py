import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.buffer)

class DQN_Agent:
    def __init__(self, state_size, hidden_size, action_size, lr, gamma, memory_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.memory = ReplayBuffer(memory_size)
        
        self.gamma = gamma
        self.action_size = action_size

        self.loss = 0.0
        
    def remember(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            action = torch.argmax(q_values).item()
            return action
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states = self.memory.sample(batch_size)
        
        current_q = self.policy_net(states)
        current_q = current_q.gather(1, actions.view(1, -1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        
        target_q = rewards + (self.gamma * next_q)
        
        self.loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        self.loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_loss(self):
        return self.loss.item() if isinstance(self.loss, torch.Tensor) else self.loss
