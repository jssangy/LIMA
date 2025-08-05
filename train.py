import torch
import argparse
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
# 이 환경들은 이제 단일 교차로를 제어하는 것을 전제로 합니다.
from gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead


def main(args):
    # --- 1. 설정 및 초기화 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 2. 환경 생성 ---
    base_env = GymEnv(prob_path=args.problem)
    env = GymWrapper(base_env, device=DEVICE)

    # --- 3. 액터-크리틱 모델 설정 ---
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

    # --- 4. 데이터 수집 및 버퍼 설정 ---
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        device=DEVICE,
        storing_device=DEVICE,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=DEVICE),
        batch_size=args.mini_batch_size,
        sampler=SamplerWithoutReplacement(),
    )

    # --- 5. 손실 함수 및 옵티마이저 설정 ---
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

    # --- 6. 훈련 루프 (튜토리얼 기반) ---
    logs = defaultdict(list)
    pbar = tqdm(total=args.total_frames)
    eval_str = ""
    total_collected_frames = 0

    for i, tensordict_data in enumerate(collector):
        # tensordict_data shape: [frames_per_batch, 1]
        total_collected_frames += tensordict_data.numel()

        for _ in range(args.num_epochs):
            keys_to_squeeze = ["action", "logits", "sample_log_prob"]
            for key in keys_to_squeeze:
                # tensordict_data[key]는 [256, 1, ...] 형태를 가집니다.
                # .squeeze(1)을 통해 [256, ...] 형태로 바꿉니다.
                tensordict_data.set(key, tensordict_data.get(key).squeeze(1))

            # GAE는 매 에포크마다 재계산 (가치 함수가 업데이트되므로)
            with torch.no_grad():
                advantage_module(tensordict_data)

            # [핵심 수정] 튜토리얼 방식: 데이터를 1차원으로 펼쳐서 버퍼에 추가
            # [256, 1] -> [256]
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            for sub_data in replay_buffer:
                # sub_data shape: [mini_batch_size]
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
            
            # 다음 에포크를 위해 버퍼 비우기
            replay_buffer.empty()

        # 로깅 및 진행 상황 업데이트
        pbar.update(tensordict_data.numel())
        reward = tensordict_data["next", "reward"].mean().item()
        logs["reward"].append(reward)
        
        if (i + 1) % 10 == 0:
            pbar.set_description(f"Iter {i+1}, Reward: {reward:.4f}")

    collector.shutdown()
    pbar.close()
    print("Training finished.")
    
    # --- 7. 모델 저장 ---
    torch.save(policy.state_dict(), 'ppo_policy.pth')
    torch.save(value_module.state_dict(), 'ppo_value.pth')
    print("Saved models to ppo_policy.pth and ppo_value.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script for DAA-CPS")
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