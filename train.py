import numpy as np
from tqdm import tqdm
import torch
import wandb

from Environment import ENV
from agent import Actor, Critic, MADDPGTrainer, ReplayBuffer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

class AGVGridVisualizer:
    def __init__(self, grid, agent_nums, goal_dict=None):
        self.grid = grid
        self.agent_nums = agent_nums
        self.agent_patches = {}
        self.goal_patches = {}
        self.colors = list(mcolors.TABLEAU_COLORS.values())  # AGV 색상 팔레트
        self.goal_dict = goal_dict if goal_dict else {}  # agent -> (x, y)
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.grid.shape[1])
        self.ax.set_ylim(0, self.grid.shape[0])
        self.ax.set_xticks(np.arange(0, self.grid.shape[1] + 1, 1))
        self.ax.set_yticks(np.arange(0, self.grid.shape[0] + 1, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        self.ax.set_title("AGV 위치 및 목표")

        # 장애물 표시
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 1:
                    self.ax.add_patch(
                        patches.Rectangle((x, y), 1, 1, color="black")
                    )

        # goal 표시
        for i, agent in enumerate(self.agent_nums):
            if agent in self.goal_dict:
                gx, gy = self.goal_dict[agent]
                goal_marker = patches.RegularPolygon(
                    (gx + 0.5, gy + 0.5), numVertices=5, radius=0.3,
                    orientation=np.pi / 2, color="green", alpha=0.6
                )
                self.goal_patches[agent] = goal_marker
                self.ax.add_patch(goal_marker)

        # AGV 초기 패치
        for i, agent in enumerate(self.agent_nums):
            color = self.colors[i % len(self.colors)]
            patch = patches.Circle((0.5, 0.5), 0.3, color=color, label=f"AGV {agent}")
            self.agent_patches[agent] = patch
            self.ax.add_patch(patch)

        self.ax.legend(loc='upper right')
        plt.ion()
        plt.show()

    def update(self, agv_states, timestep=None):
        for i, agent in enumerate(self.agent_nums):
            x, y, _ = map(int, agv_states[i])
            self.agent_patches[agent].center = (x + 0.5, y + 0.5)

        if timestep is not None:
            self.ax.set_title(f"Timestep {timestep}")
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)



# Hyperparameters
episodes = 10000
timesteps = 3000
batch_size = 128
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

visualizer = AGVGridVisualizer(env.controller.grid, agent_nums)

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

            action_logits = actors[agent](state_tensor).squeeze(0)

            # Exploration
            if np.random.rand() < epsilon:
                action = np.random.choice(act_dim)
            # Exploitation
            else:
                action = torch.argmax(action_logits).item()

            joint_action.append(action)

        # Env step joint state, joint action
        joint_next_state, reward = env.step(joint_state, joint_action)

        buffer.store(
            np.array(joint_state),
            np.array(joint_action),
            np.array(reward),
            np.array(joint_next_state)
        )

        # 출력: 매 타임스텝마다 확인용
        print(f"\nTimestep {timestep + 1}")
        for idx, agent in enumerate(agent_nums):
            print(f"Agent {agent}:")
            print(f"  State      : {joint_state[idx]}")
            print(f"  Action     : {joint_action[idx]}")
            print(f"  Reward     : {reward[idx]}")
            print(f"  Next State : {joint_next_state[idx]}")
        visualizer.update(joint_state, timestep+1)
        input("Press Enter to continue to the next timestep...")

        # Update
        if len(buffer) > batch_size and timestep % 10 == 0:
            actor_loss, critic_loss = trainer.update(buffer, batch_size)
            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)

        timestep_reward = np.sum(reward)
        episode_rewards.append(timestep_reward)
        total_reward += timestep_reward

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