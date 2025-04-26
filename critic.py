import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralCritic(nn.Module):
    def __init__(self, joint_obs_dim):
        super(CentralCritic, self).__init__()
        self.fc1 = nn.Linear(joint_obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v