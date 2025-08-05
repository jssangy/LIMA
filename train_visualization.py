import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# TorchRL 모듈 임포트
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

# 직접 만든 환경 및 분리된 모델 임포트
from gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead
from visualizer import MatplotlibVisualizer


# [추가] 상태 및 행동 해석을 위한 헬퍼 함수
def interpret_state(state_vector):
    """28차원 상태 벡터를 사람이 읽기 쉬운 문자열로 변환합니다."""
    if state_vector is None:
        return "State: None"
    
    state_vector = state_vector.flatten()
    dir_map = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    descriptions = []

    # 4개 방향 차선 정보 해석
    for i in range(4):
        start_idx = i * 6
        dir_name = dir_map[i]
        
        one_hot_target = state_vector[start_idx : start_idx + 4]
        distance = state_vector[start_idx + 4]
        deadlock = state_vector[start_idx + 5]
        
        target_dir = "None"
        if np.any(one_hot_target):
            target_idx = np.argmax(one_hot_target)
            target_dir = dir_map.get(target_idx, "Unknown")
            
        descriptions.append(
            f"  - Lane {dir_name}: Target={target_dir}, Dist={distance:.1f}, Deadlock={int(deadlock)}"
        )

    # 중앙 AGV 정보 해석
    center_one_hot = state_vector[24:28]
    center_target_dir = "None"
    if np.any(center_one_hot):
        center_target_idx = np.argmax(center_one_hot)
        center_target_dir = dir_map.get(center_target_idx, "Unknown")
    descriptions.append(f"  - Center AGV Target: {center_target_dir}")
    
    return "State:\n" + "\n".join(descriptions)

def interpret_action(action_vector):
    """5차원 행동 벡터를 사람이 읽기 쉬운 문자열로 변환합니다."""
    if action_vector is None:
        return "Action: None"
        
    action_vector = action_vector.flatten()
    # [수정] force_map을 제거하고 숫자 값을 직접 사용합니다.
    dir_map = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    # Repulsive Forces를 숫자 그대로 가져옵니다.
    forces = [int(f) for f in action_vector[:4]]
    center_move = dir_map.get(int(action_vector[4]), "Unknown")
    
    description = (
        f"Action:\n"
        f"  - Repulsive Forces: [N:{forces[0]}, E:{forces[1]}, S:{forces[2]}, W:{forces[3]}]\n"
        f"  - Center Move: {center_move}"
    )
    return description


def main(args):
    # --- 1. 설정 및 초기화 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 2. 환경 및 시각화 도구 생성 ---
    # [수정] render_mode 인자 제거, GymEnv의 내부 환경(base_env.env)을 시각화에 사용
    base_env = GymEnv(prob_path=args.problem)
    env = GymWrapper(base_env, device=DEVICE)
    visualizer = MatplotlibVisualizer()

    # --- 3. 액터-크리틱 모델 설정 (기존과 동일) ---
    state_dim = env.observation_spec["observation"].shape[-1]

    common_net = CommonNet(state_dim).to(DEVICE)
    common_operator = TensorDictModule(module=common_net, in_keys=["observation"], out_keys=["hidden"])

    policy_net = PolicyHead().to(DEVICE)
    policy_logits = TensorDictModule(module=policy_net, in_keys=["hidden"], out_keys=["logits"])

    policy_operator = ProbabilisticActor(
        module=policy_logits,
        spec=env.action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )

    value_net = ValueHead().to(DEVICE)
    value_operator = TensorDictModule(module=value_net, in_keys=["hidden"], out_keys=["state_value"])

    actor_value_module = ActorValueOperator(
        common_operator=common_operator,
        policy_operator=policy_operator,
        value_operator=value_operator,
    )
    policy = actor_value_module.get_policy_operator()
    value_module = actor_value_module.get_value_operator()

    # --- 4. 리플레이 버퍼 설정 (기존과 동일) ---
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=DEVICE),
        batch_size=args.mini_batch_size,
        sampler=SamplerWithoutReplacement(),
    )

    # --- 5. 손실 함수 및 옵티마이저 설정 (기존과 동일) ---
    advantage_module = GAE(
        gamma=args.gamma, 
        lmbda=args.lmbda, 
        value_network=value_module, 
        average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor=policy, 
        critic=value_module, 
        clip_epsilon=args.clip_epsilon, 
        entropy_coeff=args.entropy_coeff, 
        loss_critic_type="l2",
    )
    
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # --- 6. 훈련 루프 (수동 데이터 수집 및 시각화) ---
    logs = defaultdict(list)
    pbar = tqdm(total=args.total_frames)
    
    collected_frames = 0
    num_episodes = 0
    
    # 초기화
    td = env.reset()

    while collected_frames < args.total_frames:
        batch_data = []
        
        # --- [수정] SyncDataCollector 대신 수동으로 데이터 수집 ---
        for _ in range(args.frames_per_batch):
            # [시각화] 현재 상태를 렌더링하고 스페이스바 입력 대기
            # visualizer는 GymWrapper가 아닌 내부 ENV 객체에 직접 접근합니다.
            visualizer.render_and_wait(base_env.env)

            # 정책에 따라 행동 결정
            td = policy(td)
            
            # 환경에서 한 스텝 진행
            td_step = env.step(td)
            
            # 다음 스텝을 위한 데이터 준비
            next_td = td_step.get("next").clone()

            # [수정] 해석된 정보 출력
            current_state_np = td.get("observation").cpu().numpy()
            action_taken_np = td.get("action").cpu().numpy()
            reward_received = next_td.get("reward").item()
            next_state_np = next_td.get("observation").cpu().numpy()
            
            print("-" * 50)
            print(f"Time Step: {collected_frames}")
            print(interpret_state(current_state_np))
            print(interpret_action(action_taken_np))
            print(f"Reward: {reward_received:.4f}")
            print(interpret_state(next_state_np).replace("State:", "Next State:"))
            print("-" * 50)
            
            # 수집된 데이터를 텐서딕트에 저장
            td["next"] = next_td
            batch_data.append(td)
            
            td = next_td
            collected_frames += 1
            
            if td.get(("next", "done"), False):
                num_episodes += 1
                td = env.reset()

        # 수집된 데이터를 하나의 텐서딕트로 결합
        tensordict_data = torch.stack(batch_data, dim=0)

        # --- 훈련 로직 (기존과 동일) ---
        for _ in range(args.num_epochs):
            with torch.no_grad():
                advantage_module(tensordict_data)

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            for sub_data in replay_buffer:
                loss_vals = loss_module(sub_data)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            replay_buffer.empty()

        # --- 로깅 (기존과 동일) ---
        pbar.update(tensordict_data.numel())
        reward = tensordict_data["next", "reward"].mean().item()
        logs["reward"].append(reward)
        pbar.set_description(f"Frames: {collected_frames}, Reward: {reward:.4f}")

    visualizer.close() # [수정] 시각화 창 닫기
    pbar.close()
    print("Training finished.")
    
    # --- 7. 모델 저장 (기존과 동일) ---
    torch.save(policy.state_dict(), 'ppo_policy.pth')
    torch.save(value_module.state_dict(), 'ppo_value.pth')
    print("Saved models to ppo_policy.pth and ppo_value.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script for DAA-CPS with Step-by-step Visualization")
    # 기존 인자들은 train.py와 동일
    parser.add_argument('--problem', '-p', type=str, default='problems/cross/cross_1.json', help='Path to the problem file')
    parser.add_argument("--total_frames", type=int, default=500_000, help="Total frames to train for")
    parser.add_argument("--frames_per_batch", type=int, default=256, help="Frames collected per data collection phase")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch size for training updates")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train on each batch of data")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient for loss")
    
    args = parser.parse_args()
    main(args)