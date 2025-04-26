import torch
from tqdm import tqdm

from Environment import ENV
from agent import Actor
from critic import CentralCritic
from buffer import MultiAgentBuffer
from trainer import MAPPOTrainer

episodes = 1000
timesteps = 3600
gamma = 0.99
clip_ratio = 0.2
actor_lr = 1e-3
critic_lr = 1e-3

env = ENV()
agv_nums = env.controller.agv_nums

obs_dim = 2
act_dim = 5

actors = {agv: Actor(obs_dim, act_dim) for agv in agv_nums}
critic = CentralCritic(joint_obs_dim=len(agv_nums)*obs_dim)

actor_optimizers = {agv: torch.optim.Adam(actors[agv].parameters(), lr=actor_lr) for agv in agv_nums}
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

trainer = MAPPOTrainer(actors, critic, actor_optimizers, critic_optimizer, gamma, clip_ratio)

buffer = MultiAgentBuffer()

for episode in tqdm(range(episodes), desc="Episodes"):
    obs = env.reset()
    total_rewards = {agv: 0 for agv in agv_nums}

    for timestep in tqdm(range(timesteps), desc="Episode {episode}", leave=False):
        actions = {}

        for agv in agv_nums:
            obs_tensor = torch.FloatTensor(obs[agv]).unsqueeze(0) # (1, obs_dim)
            logits = actors[agv](obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
            actions[agv] = action
        
        next_obs, rewards = env.step(actions)

        buffer.store(obs, actions, rewards, next_obs)

        obs = next_obs
        
        for agv in agv_nums:
            total_rewards[agv] += rewards[agv]

    trainer.update(buffer)
    buffer.clear()

    total_reward= sum(total_rewards.values())
    print(f"Episode {episode+1}: Total reward = {total_reward}")

