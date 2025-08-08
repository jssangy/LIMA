import os
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
import wandb  # [추가]

# TorchRL 모듈 임포트
from torchrl.collectors import SyncDataCollector  # [변경]
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


def main(args):
    # --- 1. 설정 및 초기화 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # [추가] wandb 초기화
    wandb.init(
        project="DAA-CPS-PPO",  # wandb 프로젝트 이름 (원하는 대로 변경 가능)
        config=vars(args),      # 하이퍼파라미터를 wandb에 저장
        name=f"ppo_run_{wandb.util.generate_id()}" # 실행(run)의 고유 이름 생성
    )

    # --- 2. 환경 생성 ---
    def make_env():
        base_env = GymEnv(prob_path=args.problem)
        return GymWrapper(base_env, device=DEVICE)
    
    env = make_env()

    # --- 3. 액터-크리틱 모델 설정 ---
    state_dim = env.observation_spec["observation"].shape[-1]
    common_net = CommonNet(state_dim).to(DEVICE)
    common_operator = TensorDictModule(module=common_net, in_keys=["observation"], out_keys=["hidden"])
    policy_net = PolicyHead().to(DEVICE)
    policy_logits = TensorDictModule(module=policy_net, in_keys=["hidden"], out_keys=["logits"])
    policy_operator = ProbabilisticActor(
        module=policy_logits, spec=env.action_spec, in_keys=["logits"], out_keys=["action"],
        distribution_class=OneHotCategorical, return_log_prob=True, log_prob_key="sample_log_prob",
    )
    value_net = ValueHead().to(DEVICE)
    value_operator = TensorDictModule(module=value_net, in_keys=["hidden"], out_keys=["state_value"])
    actor_value_module = ActorValueOperator(
        common_operator=common_operator, policy_operator=policy_operator, value_operator=value_operator,
    )
    policy = actor_value_module.get_policy_operator()
    value_module = actor_value_module.get_value_operator()

    # --- 4. 데이터 수집 및 버퍼 설정 ---
    collector = SyncDataCollector(  # [변경]
        env,  # [변경] 단일 환경 사용
        policy, 
        frames_per_batch=args.frames_per_batch,  # [변경] num_workers 곱하기 제거
        total_frames=args.total_frames, 
        device=DEVICE, 
        storing_device=DEVICE,
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=DEVICE),  # [변경]
        batch_size=args.mini_batch_size,  # [변경] num_workers 곱하기 제거
        sampler=SamplerWithoutReplacement(),
    )

    # --- 5. 손실 함수 및 옵티마이저 설정 ---
    advantage_module = GAE(
        gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor=policy, critic=value_module, clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff, loss_critic_type="l2",
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # --- 6. 훈련 루프 ---
    logs = defaultdict(list)
    pbar = tqdm(total=args.total_frames)
    total_collected_frames = 0

    for i, tensordict_data in enumerate(collector):
        total_collected_frames += tensordict_data.numel()

        # 상태 벡터를 기반으로 교차로에 AGV가 있는 데이터만 필터링합니다.
        obs = tensordict_data.get("observation")
        is_active_mask = (
            (obs[..., 4] > 0) | 
            (obs[..., 10] > 0) | 
            (obs[..., 16] > 0) | 
            (obs[..., 22] > 0) | 
            (torch.sum(obs[..., 24:28], dim=-1) > 0)
        )
        
        processed_data = tensordict_data[is_active_mask]

        if processed_data.numel() == 0:
            pbar.update(tensordict_data.numel())
            pbar.set_description(f"Iter {i+1}, Reward: N/A (Skipped Training)")
            continue

        # --- 훈련 로직 ---
        # [추가] 로깅을 위한 손실 값 누적 변수
        total_loss_objective = 0
        total_loss_critic = 0
        total_loss_entropy = 0
        update_count = 0

        for _ in range(args.num_epochs):
            with torch.no_grad():
                advantage_module(processed_data)

            data_view = processed_data.reshape(-1)
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

                # [추가] 손실 값 누적
                total_loss_objective += loss_vals["loss_objective"].item()
                total_loss_critic += loss_vals["loss_critic"].item()
                total_loss_entropy += loss_vals["loss_entropy"].item()
                update_count += 1
            
            replay_buffer.empty()

        # --- 로깅 ---
        pbar.update(tensordict_data.numel())
        avg_reward = processed_data["next", "reward"].mean().item()
        logs["reward"].append(avg_reward)
        pbar.set_description(f"Iter {i+1}, Reward: {avg_reward:.4f}")

        # [추가] wandb에 로그 기록
        if update_count > 0:
            wandb.log({
                "reward": avg_reward,
                "loss/objective": total_loss_objective / update_count,
                "loss/critic": total_loss_critic / update_count,
                "loss/entropy": total_loss_entropy / update_count,
            }, step=total_collected_frames)


    collector.shutdown()
    pbar.close()
    print("Training finished.")
    
    # --- 7. 모델 저장 ---
    os.makedirs('checkpoint', exist_ok=True)
    torch.save(policy.state_dict(), 'checkpoint/ppo_policy.pth')
    torch.save(value_module.state_dict(), 'checkpoint/ppo_value.pth')
    print("Saved models to checkpoint/ppo_policy.pth and checkpoint/ppo_value.pth")

    # [추가] wandb 실행 종료
    wandb.finish()


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