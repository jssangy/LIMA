import random
import numpy as np
from tqdm import tqdm
import torch
import wandb

from Environment import ENV
from agent import Actor, Critic, MADDPGTrainer, ReplayBuffer

# Reproducibility
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
episodes = 50000
timesteps = 1000
batch_size = 256
gamma = 0.95
tau = 0.01
actor_lr = 0.01
critic_lr = 0.01
epsilon_start = 0.8
epsilon_end = 0.1
episode_end = 20000
epsilon = epsilon_start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_reward = -np.inf

wandb.init(
    project="DAA_CPS",  
    name=f"train3_{wandb.util.generate_id()}",
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

state_dim = 5
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
    act_dim,
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

# trainer.load_models("model/best_model_simple")

# Replay Buffer
buffer = ReplayBuffer(state_dim, num_agents, max_size=int(1e6))

# Train Loop
best_reward = -np.inf
best_episode = -1
for episode in tqdm(range(episodes)):
    env.reset()
    total_reward = 0
    episode_rewards = []
    episode_stats = {
        agent: {
            "episode_rewards": [],
            "actor_loss": [],
            "critic_loss": [],
            "q_value": [],
            "target_q_value": []
        } for agent in agent_nums
    }


    for timestep in tqdm(range(timesteps), desc=f"Episode {episode+1}", leave=False):
        joint_action = []
        joint_state = []
        joint_next_state = []

        for agent in agent_nums:
            state = env.get_state(agent)
            joint_state.append(state)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)

            # Action masking
            actors[agent].eval()
            with torch.no_grad():
                action_logits = actors[agent](state_tensor).squeeze(0)
            actors[agent].train()
            action_mask = env.valid_actions(int(state[0]), int(state[1]))
            action_mask_tensor = torch.tensor(action_mask, dtype = torch.float32, device=device)
            masked_logits = action_logits + (1 - action_mask_tensor) * (-1e9)

            # Exploration
            if np.random.rand() < epsilon:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
            # Exploitation
            else:
                action = torch.argmax(masked_logits).item()

            joint_action.append(action)

        # Env step joint state, joint action
        joint_next_state, reward, dones = env.step(joint_state, joint_action)

        bool_dones = [done in ["success", "collision"] for done in dones]

        buffer.store(
            np.array(joint_state),
            np.array(joint_action),
            np.array(reward),
            np.array(joint_next_state),
            np.array(bool_dones)
        )

        # Update
        for i, agent in enumerate(agent_nums):
            episode_stats[agent]["episode_rewards"].append(reward[i])
        
        if len(buffer) > batch_size and timestep % 10 == 0:
            stats = trainer.update(buffer, batch_size)
            for agent in agent_nums:
                episode_stats[agent]["actor_loss"].append(stats[f"{agent}/actor_loss"])
                episode_stats[agent]["critic_loss"].append(stats[f"{agent}/critic_loss"])
                episode_stats[agent]["q_value"].append(stats[f"{agent}/q_value"])
                episode_stats[agent]["target_q_value"].append(stats[f"{agent}/target_q_value"])

        timestep_reward = np.sum(reward)
        episode_rewards.append(timestep_reward)
        total_reward += timestep_reward

        if all(d == "collision" for d in dones):
            break

    epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / episode_end))

    avg_reward = np.mean(episode_rewards)
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_episode = episode + 1
        trainer.save_models(f"./checkpoint3/best_{episode+1}")

    log_data = {"Total Average Reward": avg_reward, "Timestep Duration": timestep+1, "Epsilon": epsilon}
    for agent in agent_nums:
        log_data[f"{agent}/avg_reward"] = np.mean(episode_stats[agent]["episode_rewards"])
        log_data[f"{agent}/actor_loss"] = np.mean(episode_stats[agent]["actor_loss"])
        log_data[f"{agent}/critic_loss"] = np.mean(episode_stats[agent]["critic_loss"])
        log_data[f"{agent}/q_value"] = np.mean(episode_stats[agent]["q_value"])
        log_data[f"{agent}/target_q_value"] = np.mean(episode_stats[agent]["target_q_value"])

    wandb.log(log_data, step=episode+1)

print(f"Best Model Episode {best_episode}, Average Reward = {best_reward:.2f}")