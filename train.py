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
agent_nums = env.agv_list

obs_dim = 24
act_dim = 5

actors = {agent: Actor(obs_dim, act_dim) for agent in agent_nums}
critic = CentralCritic(joint_obs_dim=len(agent_nums)*obs_dim)

actor_optimizers = {agent: torch.optim.Adam(actors[agent].parameters(), lr=actor_lr) for agent in agent_nums}
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

trainer = MAPPOTrainer(actors, critic, actor_optimizers, critic_optimizer, gamma, clip_ratio)

buffer = MultiAgentBuffer()

for episode in tqdm(range(episodes), desc="Episodes"):
    env.reset()
    total_rewards = {agent: 0 for agent in agent_nums}

    for timestep in tqdm(range(timesteps), desc="Episode {episode}", leave=False):
        actions = {}

        for agent in agent_nums:
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0) # (1, obs_dim)
            logits = actors[agent](obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
            actions[agent] = action
        
        next_obs, rewards = env.step(actions)

        buffer.store(obs, actions, rewards, next_obs)

        obs = next_obs
        
        for agent in agent_nums:
            total_rewards[agent] += rewards[agent]

    trainer.update(buffer)
    buffer.clear()

    total_reward= sum(total_rewards.values())
    print(f"Episode {episode+1}: Total reward = {total_reward}")

