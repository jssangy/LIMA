import os
import random
import numpy as np
from tqdm import tqdm
import torch
import wandb

from Environment import ENV
from agent import Actor, Critic, MADDPGTrainer, ReplayBuffer

def action_to_delta(action):
    mapping = {
        0: (0, 1),    # Up
        1: (0, -1),   # Down
        2: (1, 0),    # Right
        3: (-1, 0),   # Left
        4: (0, 0),    # Stop
    }
    return mapping[action]


def init_environment(cfg, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env = ENV(cfg)
    return env


def init_agents(agent_ids, state_dim, act_dim, actor_lr, critic_lr, device):
    actors, target_actors = {}, {}
    critics, target_critics = {}, {}
    actor_opts, critic_opts = {}, {}

    for agent in agent_ids:
        actors[agent] = Actor(state_dim, act_dim).to(device)
        target_actors[agent] = Actor(state_dim, act_dim).to(device)
        critics[agent] = Critic(state_dim, act_dim, len(agent_ids)).to(device)
        target_critics[agent] = Critic(state_dim, act_dim, len(agent_ids)).to(device)
        actor_opts[agent] = torch.optim.Adam(actors[agent].parameters(), lr=actor_lr)
        critic_opts[agent] = torch.optim.Adam(critics[agent].parameters(), lr=critic_lr)
        target_actors[agent].load_state_dict(actors[agent].state_dict())
        target_critics[agent].load_state_dict(critics[agent].state_dict())

    return actors, critics, target_actors, target_critics, actor_opts, critic_opts


def train_loop(env, actors, critics, target_actors, target_critics, actor_opts, critic_opts,
               episodes, timesteps, batch_size, gamma, tau, epsilon_start, epsilon_end, episode_end,
               state_dim, act_dim, directory, device):
    
    agent_ids = list(env.agv_list.keys())
    buffer = ReplayBuffer(state_dim, len(agent_ids), int(1e6))
    trainer = MADDPGTrainer(agent_ids, act_dim, actors, critics, target_actors, target_critics,
                            actor_opts, critic_opts, gamma, tau, device)

    best_reward = -np.inf
    best_episode = -1
    epsilon = epsilon_start

    for episode in tqdm(range(episodes)):
        env.reset()
        total_reward = 0
        episode_rewards = []
        episode_stats = {agent: {"episode_rewards": [], "actor_loss": [], "critic_loss": [],
                                 "q_value": [], "target_q_value": []} for agent in agent_ids}

        for timestep in tqdm(range(timesteps), desc=f"Episode {episode+1}", leave=False):
            joint_action, joint_state, predicted_next_positions = [], [], {}

            for idx, agent in enumerate(agent_ids):
                state = env.get_state(agent)
                joint_state.append(state)
                x, y = map(int, state[:2])
                
                occupied = set(predicted_next_positions.values())
                for later_agent in agent_ids[idx+1:]:
                    lx, ly = map(int, env.get_state(later_agent)[:2])
                    occupied.add((lx, ly))
                action_mask = env.valid_actions(x, y, occupied)
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                actors[agent].eval()
                with torch.no_grad():
                    logits = actors[agent](state_tensor).squeeze(0)
                actors[agent].train()
                masked_logits = logits + (1 - action_mask_tensor) * (-1e9)

                if np.random.rand() < epsilon:
                    valid_actions = np.where(action_mask)[0]
                    action = np.random.choice(valid_actions)
                else:
                    action = torch.argmax(masked_logits).item()

                joint_action.append(action)
                dx, dy = action_to_delta(action)
                predicted_next_positions[agent] = (x + dx, y + dy)

            joint_next_state, reward, dones = env.step(joint_state, joint_action)
            bool_dones = [done in ["success", "collision"] for done in dones]

            buffer.store(np.array(joint_state), np.array(joint_action), np.array(reward),
                         np.array(joint_next_state), np.array(bool_dones))

            for i, agent in enumerate(agent_ids):
                episode_stats[agent]["episode_rewards"].append(reward[i])

            if len(buffer) > batch_size and timestep % 10 == 0:
                stats = trainer.update(buffer, batch_size)
                for agent in agent_ids:
                    episode_stats[agent]["actor_loss"].append(stats[f"{agent}/actor_loss"])
                    episode_stats[agent]["critic_loss"].append(stats[f"{agent}/critic_loss"])
                    episode_stats[agent]["q_value"].append(stats[f"{agent}/q_value"])
                    episode_stats[agent]["target_q_value"].append(stats[f"{agent}/target_q_value"])

            timestep_reward = np.sum(reward)
            episode_rewards.append(timestep_reward)
            total_reward += timestep_reward

            # if all(d == "collision" for d in dones):
            #     break

        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / episode_end))
        avg_reward = np.mean(episode_rewards)

        if avg_reward > best_reward and all(env.task_done_flags.values()):
            best_reward = avg_reward
            best_episode = episode + 1
            os.makedirs(directory, exist_ok=True)
            trainer.save_models(os.path.join(directory, f'best_{episode+1}'))

        log_data = {"Total Average Reward": avg_reward, "Timestep Duration": timestep+1, "Epsilon": epsilon}
        for agent in agent_ids:
            log_data[f"{agent}/avg_reward"] = np.mean(episode_stats[agent]["episode_rewards"])
            log_data[f"{agent}/actor_loss"] = np.mean(episode_stats[agent]["actor_loss"])
            log_data[f"{agent}/critic_loss"] = np.mean(episode_stats[agent]["critic_loss"])
            log_data[f"{agent}/q_value"] = np.mean(episode_stats[agent]["q_value"])
            log_data[f"{agent}/target_q_value"] = np.mean(episode_stats[agent]["target_q_value"])
        wandb.log(log_data, step=episode+1)

    print(f"Best Model Episode {best_episode}, Average Reward = {best_reward:.2f}")

def main():
    device = torch.device("cuda")
    env = init_environment(0)
    agent_ids = list(env.agv_list.keys())
    actors, critics, target_actors, target_critics, actor_opts, critic_opts = init_agents(
        agent_ids, state_dim=5, act_dim=5, actor_lr=0.01, critic_lr=0.01, device=device
    )
    wandb.init(project="DAA_CPS", name=f"EZware_{wandb.util.generate_id()}")

    train_loop(
        env, actors, critics, target_actors, target_critics,
        actor_opts, critic_opts,
        episodes=50000, timesteps=1000, batch_size=256,
        gamma=0.95, tau=0.01,
        epsilon_start=0.8, epsilon_end=0.1, episode_end=25000,
        state_dim=5, act_dim=5, directory="./checkpoint",
        device=device
    )

if __name__ == "__main__":
    main()