import torch

class MAPPOTrainer:
    def __init__(self, actor_dict, critic, actor_optimizer_dict, critic_optimizer, gamma=0.99, clip_ratio=0.2):
        self.actor_dict = actor_dict
        self.critic = critic
        self.actor_optimizer_dict = actor_optimizer_dict
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio

    def update(self, buffer):
        obs, actions, rewards, next_obs = buffer.get_all()

        obs = torch.FloatTensor(obs)
        next_obs = torch.FloatTensor(next_obs)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # Joint Observations (concatenate across agents)
        joint_obs = obs.view(obs.shape[0], -1)
        joint_next_obs = next_obs.view(next_obs.shape[0], -1)

        # Compute advantages
        values = self.critic(joint_obs).squeeze()
        next_values = self.critic(joint_next_obs).squeeze()
        targets = rewards.sum(dim=1) + self.gamma * next_values
        advantages = targets - values

        # Update Critic
        critic_loss = (advantages ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update each Actor
        for agent_id in self.actor_dict.keys():
            actor = self.actor_dict[agent_id]
            optimizer = self.actor_optimizer_dict[agent_id]

            logits = actor(obs[:, agent_id, :])
            log_probs = torch.log_softmax(logits, dim=-1)
            action_taken_log_probs = log_probs.gather(1, actions[:, agent_id].unsqueeze(1)).squeeze()

            with torch.no_grad():
                old_log_probs = action_taken_log_probs

            ratios = torch.exp(action_taken_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
