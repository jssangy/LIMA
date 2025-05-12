import numpy as np
from tqdm import tqdm
import torch
import wandb

from Environment import ENV
from agent import Actor, Critic, MADDPGTrainer, ReplayBuffer

# Hyperparameters
episodes = 10000
timesteps = 3000
batch_size = 128
gamma = 0.95
tau = 0.01
actor_lr = 1e-3
critic_lr = 1e-3
epsilon_start = 0.8
epsilon_final = 0
epsilon = epsilon_start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_reward = -np.inf
num_gpus = torch.cuda.device_count()

wandb.init(
    project="DAA_CPS",  
    name=f"train_run_{wandb.util.generate_id()}",
    config={
        "episodes": episodes,
        "timesteps": timesteps,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "epsilon": epsilon
    }
)

# Environment
env = ENV()

agent_nums = list(env.agv_list.keys())
num_agents = len(agent_nums)

state_dim = 3
act_dim = 5

# Actor, Critic Network
actors = {}
target_actors = {}
critics = {}
target_critics = {}
actor_opts = {}
critic_opts = {}
for i, agent in enumerate(agent_nums):
    actors[agent] = Actor(state_dim, act_dim).to(device)
    target_actors[agent] = Actor(state_dim, act_dim).to(device)
    critics[agent] = Critic(state_dim, act_dim, num_agents).to(device)
    target_critics[agent] = Critic(state_dim, act_dim, num_agents).to(device)

    actor_opts[agent] = torch.optim.Adam(actors[agent].parameters(), lr=actor_lr)
    critic_opts[agent] = torch.optim.Adam(critics[agent].parameters(), lr=critic_lr)

# Optimizers
actor_opts = {agent: torch.optim.Adam(actors[agent].parameters(), lr=actor_lr) for agent in agent_nums}
critic_opts = {agent: torch.optim.Adam(critics[agent].parameters(), lr=critic_lr) for agent in agent_nums}

# Target Network initialization
for agent in agent_nums:
    target_actors[agent].load_state_dict(actors[agent].state_dict())
    target_critics[agent].load_state_dict(critics[agent].state_dict())

# MADDPG Trainer
trainer = MADDPGTrainer(
    agent_nums,
    actor_dict=actors,
    critic_dict=critics,
    actor_target_dict=target_actors,
    critic_target_dict=target_critics,
    actor_opt_dict=actor_opts,
    critic_opt_dict=critic_opts,
    gamma=gamma,
    tau=tau,
    device=device
)

# Replay Buffer
buffer = ReplayBuffer(state_dim, num_agents, max_size=int(1e6))

# Train Loop
for episode in range(episodes):
    env.reset()
    total_reward = 0
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []

    for timestep in tqdm(range(timesteps), desc=f"Episode {episode+1}", leave=False):
        joint_action = []
        joint_state = []
        joint_next_state = []

        for agent in agent_nums:
            state = env.get_state(agent)
            joint_state.append(state)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)

            # Action masking
            action_mask = env.valid_actions(int(state[0]), int(state[1]))
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
            action_logits = actors[agent](state_tensor).squeeze(0)  # (act_dim,)
            masked_logits = action_logits + (1 - action_mask_tensor) * (-1e9)

            # Exploration
            if np.random.rand() < epsilon:
                valid_actions = np.where(np.array(action_mask) == 1)[0]
                action = np.random.choice(valid_actions)
            # Exploitation
            else:
                action_probs = torch.softmax(masked_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()

            joint_action.append(action)

        # Env step joint state, joint action
        joint_next_state, reward = env.step(joint_state, joint_action)

        buffer.store(
            np.array(joint_state),
            np.array(joint_action),
            np.array(reward),
            np.array(joint_next_state)
        )

        # Update
        if len(buffer) > batch_size and timestep % 10 == 0:
            actor_loss, critic_loss = trainer.update(buffer, batch_size)
            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)

        timestep_reward = np.sum(reward)
        episode_rewards.append(timestep_reward)
        total_reward += timestep_reward

    epsilon = max(epsilon_final, epsilon_start - (epsilon_start - epsilon_final) * (episode / episodes))

    avg_reward = np.mean(episode_rewards)
    avg_actor_loss = np.mean(episode_actor_losses)
    avg_critic_loss = np.mean(episode_critic_losses)
    if total_reward > best_reward:
        best_reward = total_reward
        trainer.save_models(f"./checkpoints/best_model")
        print(f"Best Model Episode {episode+1}, Total Reward = {total_reward}, Avg timestep Reward = {avg_reward:.2f}")
        
    if (episode+1) % 100 == 0:
        trainer.save_models(f"./checkpoints/episode_{episode+1}")

    wandb.log({
        "Total Reward": total_reward,
        "Average Timestep Reward": avg_reward,
        "Average Actor Loss": avg_actor_loss,
        "Average Critic Loss": avg_critic_loss
    }, step=episode+1)