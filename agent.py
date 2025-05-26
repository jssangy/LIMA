import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor for Discrete Action
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.bn = nn.BatchNorm1d(obs_dim)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(0)

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(self.bn(x)))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# Critic: (state, action) concat
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents):
        super(Critic, self).__init__()
        input_dim = num_agents * (obs_dim + act_dim)

        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(self.bn(x)))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v


class MADDPGTrainer:
    def __init__(self, agent_nums, act_dim, actor_dict, critic_dict, actor_target_dict, critic_target_dict,
                 actor_opt_dict, critic_opt_dict, gamma=0.95, tau=0.01, device='cuda'):
        self.agent_nums = agent_nums
        self.action_space = act_dim
        self.actor_dict = actor_dict
        self.critic_dict = critic_dict
        self.actor_target_dict = actor_target_dict
        self.critic_target_dict = critic_target_dict
        self.actor_opt_dict = actor_opt_dict
        self.critic_opt_dict = critic_opt_dict
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def update(self, buffer, batch_size):
        state, actions, rewards, next_state, dones = buffer.sample(batch_size)
        state = state.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_state = next_state.to(self.device)
        dones = dones.to(self.device)

        stats = {}

        for agent in self.agent_nums:

            if agent not in ["D", "E"]:
                continue
            
            agent_idx = self.agent_nums.index(agent)

            # Critic Update
            joint_state = state.view(batch_size, -1)
            joint_action = F.one_hot(actions.long(), num_classes=self.action_space).float().view(batch_size, -1)
            joint_next_state = next_state.view(batch_size, -1)

            with torch.no_grad():
                target_next_actions = []
                for other_agent in self.agent_nums:
                    other_idx = self.agent_nums.index(other_agent)
                    next_state_agent = next_state[:, other_idx, :]
                    logits = self.actor_target_dict[other_agent](next_state_agent)
                    action_index = torch.argmax(logits, dim=-1)
                    target_action = F.one_hot(action_index, num_classes=self.action_space).float()
                    target_next_actions.append(target_action)
                target_next_actions = torch.cat(target_next_actions, dim=-1)

                target_q_value = self.critic_target_dict[agent](joint_next_state, target_next_actions).squeeze()
                target_q = rewards[:, agent_idx] + self.gamma * (1 - dones[:, agent_idx]) * target_q_value

            current_q = self.critic_dict[agent](joint_state, joint_action).squeeze()

            critic_loss = F.mse_loss(current_q, target_q)
            self.critic_opt_dict[agent].zero_grad()
            critic_loss.backward()
            self.critic_opt_dict[agent].step()

            # Actor Update
            predicted_actions = []
            for other_agent in self.agent_nums:
                other_idx = self.agent_nums.index(other_agent)
                state_agent = state[:, other_idx, :]
                logits = self.actor_dict[other_agent](state_agent)
                if other_agent == agent:
                    action = F.gumbel_softmax(logits, hard=True)
                    self_logits = logits
                else:
                    action_index = torch.argmax(logits, dim=-1)
                    action = F.one_hot(action_index, num_classes=self.action_space).float().detach()
                predicted_actions.append(action)

            predicted_actions = torch.cat(predicted_actions, dim=-1)
            actor_loss = -self.critic_dict[agent](joint_state, predicted_actions).mean()
            actor_loss += (self_logits ** 2).mean() * 1e-3

            self.actor_opt_dict[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_dict[agent].parameters(), 0.5)
            self.actor_opt_dict[agent].step()

            # Target Update
            self.soft_update(self.actor_dict[agent], self.actor_target_dict[agent])
            self.soft_update(self.critic_dict[agent], self.critic_target_dict[agent])

            stats[f"{agent}/critic_loss"] = critic_loss.item()
            stats[f"{agent}/actor_loss"] = actor_loss.item()
            stats[f"{agent}/q_value"] = current_q.mean().item()
            stats[f"{agent}/target_q_value"] = target_q.mean().item()

        return stats

    def soft_update(self, online_net, target_net):
        for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        for agent in self.agent_nums:
            torch.save(self.actor_dict[agent].state_dict(), os.path.join(path, f"actor_{agent}.pth"))
            torch.save(self.critic_dict[agent].state_dict(), os.path.join(path, f"critic_{agent}.pth"))
            torch.save(self.actor_target_dict[agent].state_dict(), os.path.join(path, f"target_actor_{agent}.pth"))
            torch.save(self.critic_target_dict[agent].state_dict(), os.path.join(path, f"target_critic_{agent}.pth"))

    def load_models(self, path):
        for agent in self.agent_nums:
            self.actor_dict[agent].load_state_dict(torch.load(os.path.join(path, f"best_model_{agent}/actor_{agent}.pth")))
            # self.critic_dict[agent].load_state_dict(torch.load(os.path.join(path, f"critic_{agent}.pth")))
            self.actor_target_dict[agent].load_state_dict(torch.load(os.path.join(path, f"best_model_{agent}/target_actor_{agent}.pth")))
            # self.critic_target_dict[agent].load_state_dict(torch.load(os.path.join(path, f"target_critic_{agent}.pth")))


class ReplayBuffer:
    def __init__(self, obs_dim, num_agents, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs_buf = np.zeros((max_size, num_agents, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, num_agents, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((max_size, num_agents), dtype=np.int32)
        self.rewards_buf = np.zeros((max_size, num_agents), dtype=np.float32)
        self.dones_buf = np.zeros((max_size, num_agents), dtype=np.float32)

    def store(self, obs, action, reward, next_obs, dones):
        self.obs_buf[self.ptr] = obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.dones_buf[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch_obs = torch.FloatTensor(self.obs_buf[idxs])
        batch_actions = torch.LongTensor(self.actions_buf[idxs])
        batch_rewards = torch.FloatTensor(self.rewards_buf[idxs])
        batch_next_obs = torch.FloatTensor(self.next_obs_buf[idxs])
        batch_dones = torch.FloatTensor(self.dones_buf[idxs])

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones

    def __len__(self):
        return self.size

