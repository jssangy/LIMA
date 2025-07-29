import torch
import torch.optim as optim
import numpy as np
import argparse
from collections import deque

from Environment import ENV
from model import ActorCritic

def ppo_update(model, optimizer, memory, gamma, lam, clip_epsilon, entropy_coef, num_epochs, batch_size, device):
    """
    메모리에 저장된 데이터로 PPO 업데이트를 수행합니다.
    """
    # 1. 데이터 추출 및 텐서 변환
    states, actions, log_probs_old, rewards, values_old, masks = zip(*memory)
    
    # torch.cat 대신 torch.stack을 사용하여 개별 샘플들을 배치로 만듭니다.
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    log_probs_old = torch.stack(log_probs_old).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    values_old = torch.stack(values_old).to(device).squeeze()
    masks = torch.tensor(masks, dtype=torch.float32).to(device)

    # 2. GAE (Generalized Advantage Estimation) 및 Returns 계산
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    with torch.no_grad():
        # 마지막 상태의 가치 추정
        next_value = model(states[-1].unsqueeze(0))[1].squeeze()

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * masks[i] - values_old[i]
        gae = delta + gamma * lam * masks[i] * gae
        next_value = values_old[i]
        returns[i] = gae + values_old[i]
    
    advantages = returns - values_old
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. 모델 업데이트 (미니배치)
    # 데이터셋을 섞기 위한 인덱스
    indices = np.arange(len(states))
    for _ in range(num_epochs):
        np.random.shuffle(indices) # 매 에포크마다 데이터를 섞어줌
        for i in range(0, len(states), batch_size):
            # 미니배치 인덱스
            mb_indices = indices[i:i+batch_size]
            
            # 미니배치 슬라이싱
            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_log_probs_old = log_probs_old[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # 새 로짓, 가치, 엔트로피 계산
            (rep_logits, cen_logits), new_values = model(mb_states)
            new_values = new_values.squeeze()

            rep_dist = torch.distributions.Categorical(logits=rep_logits)
            cen_dist = torch.distributions.Categorical(logits=cen_logits)

            # 로그 확률 계산
            rep_log_probs = rep_dist.log_prob(mb_actions[:, :4]).sum(dim=1)
            cen_log_probs = cen_dist.log_prob(mb_actions[:, 4])
            new_log_probs = rep_log_probs + cen_log_probs
            
            # 엔트로피 계산
            entropy = (rep_dist.entropy().sum(dim=1) + cen_dist.entropy()).mean()

            # PPO 손실 계산
            ratio = (new_log_probs - mb_log_probs_old).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (mb_returns - new_values).pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            # 업데이트
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

def calculate_reward(env_info, prev_throughput):
    """처리량 변화를 기반으로 보상을 계산합니다."""
    current_throughput = env_info[0]
    reward = current_throughput - prev_throughput
    return reward, current_throughput

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p', type=str, default='problems/cross/cross_1.json', help='Path to the problem file')
    args = parser.parse_args()

    # --- 1. 하이퍼파라미터 및 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TIMESTEPS = 1_000_000
    ROLLOUT_SIZE = 2048
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    LR = 3e-4

    # --- 2. 환경 및 모델 초기화 ---
    print("Loading environment and model...")
    env = ENV(args.problem)
    model = ActorCritic(state_dim=29).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # --- 3. 훈련 루프 ---
    print(f"Training starts on {DEVICE}...")
    timestep = 0
    prev_throughput = 0
    
    while timestep < MAX_TIMESTEPS:
        memory = []
        
        for _ in range(ROLLOUT_SIZE):
            timestep += 1
            
            active_intersections = env.get_active_intersections()
            actions_to_execute = {}
            step_memory = []
            
            if active_intersections:
                states = [inter.get_state() for inter in active_intersections]
                states_tensor = torch.FloatTensor(np.array(states)).to(DEVICE)
                
                with torch.no_grad():
                    (rep_logits, cen_logits), values = model(states_tensor)
                
                rep_dist = torch.distributions.Categorical(logits=rep_logits)
                rep_actions = rep_dist.sample()
                
                cen_dist = torch.distributions.Categorical(logits=cen_logits)
                cen_actions = cen_dist.sample()
                
                actions = torch.cat([rep_actions, cen_actions.unsqueeze(1)], dim=1)
                
                rep_log_probs = rep_dist.log_prob(rep_actions).sum(dim=1)
                cen_log_probs = cen_dist.log_prob(cen_actions)
                log_probs = rep_log_probs + cen_log_probs

                for i, intersection in enumerate(active_intersections):
                    actions_to_execute[intersection.id] = actions[i].cpu().numpy()
                    step_memory.append([states_tensor[i], actions[i], log_probs[i], values[i]])

            env_info = env.step(actions_to_execute)
            reward, prev_throughput = calculate_reward(env_info, prev_throughput)
            
            for state, action, log_prob, value in step_memory:
                memory.append([state, action, log_prob, reward, value, 1.0])

        # --- 4. PPO 업데이트 ---
        if memory:
            ppo_update(model, optimizer, memory, GAMMA, LAMBDA, CLIP_EPSILON, ENTROPY_COEF, NUM_EPOCHS, BATCH_SIZE, DEVICE)
            print(f"Timestep: {timestep}, Throughput: {prev_throughput:.2f}, Last Reward: {reward:.4f}, Updated with {len(memory)} samples.")

    # --- 5. 모델 저장 ---
    print("Training finished.")
    torch.save(model.state_dict(), 'ppo_intersection_model.pth')