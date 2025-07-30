import torch
import argparse
from tqdm import tqdm
import time
import numpy as np

# TorchRL 모듈 임포트
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

# 직접 만든 환경, 모델, 시각화 도구 임포트
from gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead
from visualizer import MatplotlibVisualizer

def main(args):
    # --- 1. 설정 및 초기화 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 2. 환경 및 시각화 도구 생성 ---
    base_env = GymEnv(prob_path=args.problem)
    env = GymWrapper(base_env, device=DEVICE)
    visualizer = MatplotlibVisualizer()

    # --- 3. 모델, 정책, 가치 함수 정의 (train.py와 동일) ---
    state_dim = env.observation_spec["observation"].shape[-1]
    common_net = CommonNet(state_dim).to(DEVICE)
    common_operator = TensorDictModule(module=common_net, in_keys=["observation"], out_keys=["hidden"])
    policy_net = PolicyHead().to(DEVICE)
    policy_logits = TensorDictModule(module=policy_net, in_keys=["hidden"], out_keys=["logits"])
    policy_operator = ProbabilisticActor(
        module=policy_logits, spec=env.action_spec, in_keys=["logits"], out_keys=["action"],
        distribution_class=OneHotCategorical, return_log_prob=True
    )
    value_net = ValueHead().to(DEVICE)
    value_operator = TensorDictModule(module=value_net, in_keys=["hidden"], out_keys=["state_value"])
    actor_value_module = ActorValueOperator(
        common_operator=common_operator, policy_operator=policy_operator, value_operator=value_operator
    )
    policy = actor_value_module.get_policy_operator()
    value_module = actor_value_module.get_value_operator()

    # --- 4. 리플레이 버퍼 설정 ---
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch),
        batch_size=args.mini_batch_size,
    )

    # --- 5. 손실 함수 및 옵티마이저 설정 (train.py와 동일) ---
    advantage_module = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True)
    loss_module = ClipPPOLoss(actor=policy, critic=value_module, clip_epsilon=args.clip_epsilon, entropy_coeff=args.entropy_coeff, loss_critic_type="l2")
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # --- 6. 시각화를 포함한 훈련 루프 ---
    total_collected_frames = 0
    pbar = tqdm(total=args.total_frames)
    
    # Collector 대신 수동 루프 사용
    for i in range(args.total_frames // args.frames_per_batch):
        # 데이터 수집 (롤아웃)
        rollout_data = []
        td = env.reset()
        for _ in range(args.frames_per_batch):
            # 시각화
            visualizer.render_and_wait(base_env.env) # GymWrapper 내부의 원본 env 전달

            # 액션 샘플링
            td = policy(td)
            
            # 환경 스텝
            td_next = env.step(td)

            # --- 디버깅 정보 출력 ---
            state_np = td['observation'].cpu().numpy()
            action_np = td['action'].cpu().numpy()
            # 보상(reward)은 'next' 키 아래에 저장됩니다.
            reward_val = td['next', 'reward'].item()

            print("\n" + "="*20 + f" Timestep {base_env.env.time} " + "="*20)
            # 소수점 2자리까지 반올림하여 출력
            print(f"State : {np.round(state_np, 2)}")
            print(f"Action: {action_np.astype(int)}") # Action은 정수
            print(f"Reward: {reward_val:.4f}")
            print("="*52)
            # --- 디버깅 정보 출력 끝 ---
            
            # 데이터 저장
            rollout_data.append(td.clone())
            
            td = td_next
            total_collected_frames += 1
            pbar.update(1)

        # 수집된 데이터를 하나의 TensorDict로 합침
        tensordict_data = torch.stack(rollout_data, dim=0)

        # GAE 계산
        with torch.no_grad():
            advantage_module(tensordict_data)

        # 리플레이 버퍼에 추가
        replay_buffer.add(tensordict_data.view(-1))

        # PPO 업데이트
        for _ in range(args.num_epochs):
            for sub_data in replay_buffer:
                loss_vals = loss_module(sub_data)
                loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        if (i+1) % 10 == 0:
            print(f"\nIter {i+1}, Frames {total_collected_frames}: loss={loss.item():.4f}")

    pbar.close()
    visualizer.close()
    print("Training finished.")
    
    # --- 7. 모델 저장 ---
    torch.save(policy.state_dict(), 'ppo_policy_viz.pth')
    torch.save(value_module.state_dict(), 'ppo_value_viz.pth')
    print("Saved models to ppo_policy_viz.pth and ppo_value_viz.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Visualized Training Script")
    parser.add_argument('--problem', '-p', type=str, default='problems/cross/cross_1.json', help='Path to the problem file')
    parser.add_argument("--total_frames", type=int, default=50_000, help="Total frames to train for")
    parser.add_argument("--frames_per_batch", type=int, default=512, help="Frames collected per data collection phase")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch size for training updates")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train on each batch of data")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient for loss")
    
    args = parser.parse_args()
    main(args)