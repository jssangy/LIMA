import numpy as np
from tqdm import tqdm
import torch
import wandb

from Environment import ENV
from agent import Actor, Critic, MADDPGTrainer, ReplayBuffer

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rgb_to_mpl(rgb):
    return tuple(v / 255.0 for v in rgb)

class AGVGridVisualizer:
    def __init__(self, grid, agent_nums, env):
        self.grid = grid
        self.agent_nums = agent_nums
        self.env = env 
        self.agent_patches = {}
        self.goal_patches = {}
        self.colors = [
            rgb_to_mpl((255, 0, 0)),     # Red
            rgb_to_mpl((0, 255, 0)),     # Green
            rgb_to_mpl((0, 0, 255)),     # Blue
            rgb_to_mpl((255, 255, 0)),   # Yellow
            rgb_to_mpl((255, 0, 255)),   # Pink
        ]
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.grid.shape[1])
        self.ax.set_xticks(np.arange(0, self.grid.shape[1] + 1, 1))
        self.ax.set_yticks(np.arange(0, self.grid.shape[0] + 1, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        self.ax.set_title("AGV position and goal")

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 1:
                    self.ax.add_patch(
                        patches.Rectangle((x, y), 1, 1, color="black")
                    )

        for i, agent in enumerate(self.agent_nums):
            color = self.colors[i % len(self.colors)]
            patch = patches.Circle((0.5, 0.5), 0.3, color=color)
            self.agent_patches[agent] = patch
            self.ax.add_patch(patch)

        plt.ion()
        plt.show()

    def update(self, agv_states, episode, timestep=None):
        for i, agent in enumerate(self.agent_nums):
            x, y, _, __, ___ = map(int, agv_states[i])
            self.agent_patches[agent].center = (x + 0.5, y + 0.5)

        for marker in self.goal_patches.values():
            marker.remove()
        self.goal_patches = {}

        for i, agent in enumerate(self.agent_nums):
            color = self.colors[i % len(self.colors)]
            goal_pos = self.env.agv_list[agent].goal
            if goal_pos:
                gx, gy = map(int, goal_pos)
                marker = patches.RegularPolygon(
                    (gx + 0.5, gy + 0.5), numVertices=5, radius=0.3,
                    orientation=np.pi / 2, color=color, alpha=0.6
                )
                self.goal_patches[agent] = marker
                self.ax.add_patch(marker)

        if timestep is not None:
            self.ax.set_title(f"Episode {episode}, Timestep {timestep}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Hyperparameters
episodes = 1000
timesteps = 3000
batch_size = 32
gamma = 0.95
tau = 0.01
actor_lr = 0.01
critic_lr = 0.01
epsilon = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_reward = -np.inf

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

# Replay Buffer
buffer = ReplayBuffer(state_dim, num_agents, max_size=int(1e6))

visualizer = AGVGridVisualizer(env.controller.grid, agent_nums, env)

# Train Loop
best_reward = -np.inf
best_episode = -1
for episode in range(episodes):
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


    for timestep in range(timesteps):
        joint_action = []
        joint_state = []
        joint_next_state = []
        explore_fig = []

        for agent in agent_nums:
            state = env.get_state(agent)
            joint_state.append(state)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)

            # Action masking
            actors[agent].eval()
            with torch.no_grad():
                action_logits = actors[agent](state_tensor).squeeze(0)
            actors[agent].train()

            # Exploration
            if np.random.rand() < epsilon:
                action = np.random.choice(act_dim)
                explore_fig.append(True)
            # Exploitation
            else:
                action = torch.argmax(action_logits).item()
                explore_fig.append(False)

            joint_action.append(action)

        # Env step joint state, joint action
        joint_next_state, reward, dones = env.step(joint_state, joint_action)

        buffer.store(
            np.array(joint_state),
            np.array(joint_action),
            np.array(reward),
            np.array(joint_next_state),
            np.array(dones)
        )

        action_list = ['up', 'down', 'right', 'left', 'stop']
        print(f"\nEpisode {episode}, Timestep {timestep + 1}")
        for idx, agent in enumerate(agent_nums):
            print(f"Agent {agent}:")
            print(f"  Prev State : {joint_state[idx]}")
            print(f"  Explore    : {explore_fig[idx]}")
            print(f"  Action     : {action_list[joint_action[idx]]}")
            print(f"  Reward     : {reward[idx]}")
            print(f"  Cur State  : {joint_next_state[idx]}")
            print(f"  Dones      : {dones[idx]}")
        visualizer.update(joint_state, episode+1, timestep+1)
        # input("Press Enter to continue to the next timestep...")

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

        if np.any(dones):
            break

    avg_reward = np.mean(episode_rewards)
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_episode = episode + 1
        trainer.save_models(f"./checkpoints/best_model")
        
    if (episode+1) % 1000 == 0:
        trainer.save_models(f"./checkpoints/episode_{episode+1}")

    log_data = {"Total Average Reward": avg_reward}
    for agent in agent_nums:
        log_data[f"{agent}/avg_reward"] = np.mean(episode_stats[agent]["episode_rewards"])
        log_data[f"{agent}/actor_loss"] = np.mean(episode_stats[agent]["actor_loss"])
        log_data[f"{agent}/critic_loss"] = np.mean(episode_stats[agent]["critic_loss"])
        log_data[f"{agent}/q_value"] = np.mean(episode_stats[agent]["q_value"])
        log_data[f"{agent}/target_q_value"] = np.mean(episode_stats[agent]["target_q_value"])

    wandb.log(log_data, step=episode+1)

print(f"Best Model Episode {best_episode}, Average Reward = {best_reward}")