import os
import time
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

from module.Environment import GymEnv
from module.model import GNNBody, Actor, Critic


def set_seed(seed):
    """모든 랜덤 시드를 고정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ All random seeds set to: {seed}")


def main(args):
    # --- 0. 시드 설정 및 장치 선택 ---
    if args.seed is not None: set_seed(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- wandb 및 저장 경로 설정 ---
    exp_name = f"lr{args.lr:.0e}_gm{args.gamma}_lam{args.lmbda}_clip{args.clip_epsilon}_ent{args.entropy_coeff:.0e}"
    wandb.init(project="GNN_EventPPO", config=vars(args), name=exp_name)
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. 환경 생성 (단일 에이전트) ---
    # GymWrapper만 사용하여 표준 단일 에이전트 환경을 래핑합니다.
    # obs_keys를 명시하여 관측값들을 "observation" 키 아래에 그룹핑합니다.
    base_env = GymEnv(prob_path=args.problem)
    env = GymWrapper(base_env, device=DEVICE)

    # --- 2. 액터-크리틱 모델 설정 ---
    state_dim = env.observation_spec["nodes"].shape[-1]
    
    # 모델 초기화 시에도 시드가 적용됨
    gnn_body = GNNBody(state_dim).to(DEVICE)
    common_operator = TensorDictModule(
        module=gnn_body, in_keys=["nodes", "edge_index", "active_nodes"], out_keys=["node_embeddings"]
    )
    policy_net = Actor().to(DEVICE)
    policy_logits = TensorDictModule(
        module=policy_net, in_keys=["node_embeddings"], out_keys=["logits"]
    )
    policy_operator = ProbabilisticActor(
        module=policy_logits,
        spec=env.action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        distribution_kwargs={"mask": ["action_mask"]},
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    value_net = Critic().to(DEVICE)
    value_operator = TensorDictModule(
        module=value_net, in_keys=["node_embeddings", "batch"], out_keys=["state_value"]
    )
    actor_value_module = ActorValueOperator(
        common_operator=common_operator,
        policy_operator=policy_operator,
        value_operator=value_operator,
    )
    policy = actor_value_module.get_policy_operator()
    value_module = actor_value_module.get_value_operator()

    # --- 3. 데이터 수집 및 버퍼 설정 ---
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

    # --- 4. 손실 함수 및 옵티마이저 설정 ---
    advantage_module = GAE(
        gamma=args.gamma, 
        lmbda=args.lmbda, 
        value_network=value_module, 
        average_gae=True, 
        deactivate_vmap=True,
    )
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        loss_critic_type="l2",
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # --- 5. 훈련 루프 ---
    pbar = tqdm(total=args.total_frames)
    total_collected_frames = 0

    for i, tensordict_data in enumerate(collector):
        total_collected_frames += tensordict_data.numel()
        pbar.update(tensordict_data.numel())

        # --- [핵심 수정] 관측 기반으로 이벤트 궤적 필터링 ---
        # 1. 'nodes' 텐서의 합이 0보다 큰지 여부를 나타내는 boolean 마스크 생성
        # tensordict_data["nodes"]의 모양은 [Batch, Time, N_MAX, STATE_DIM]
        obs_nodes = tensordict_data.get("nodes")
        is_event_step = obs_nodes.abs().sum(dim=(-1, -2)) > 0 # 노드와 특징 차원을 합산

        # 2. boolean 마스크를 사용하여 이벤트가 발생한 스텝의 데이터만 필터링
        filtered_data = tensordict_data[is_event_step]
        # --- 필터링 종료 ---
        
        # GAE 계산 (필터링된 고품질 데이터에 대해서만 수행)
        with torch.no_grad():
            advantage_module(filtered_data)

        data_view = filtered_data.reshape(-1)
        replay_buffer.extend(data_view)

        # 학습 진행 (기존과 동일)
        for _ in range(args.num_epochs):
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

        # 로깅 (필터링된 데이터 기준)
        avg_reward = filtered_data.get(("next", "reward")).mean().item()
        wandb.log({
            "reward/mean_filtered": avg_reward,
            "loss/objective": loss_vals["loss_objective"].item(),
            "loss/critic": loss_vals["loss_critic"].item(),
            "loss/entropy": loss_vals["loss_entropy"].item()
        }, step=total_collected_frames)
        pbar.set_description(f"Iter {i+1}, Filtered Reward: {avg_reward:.4f}")
    
    collector.shutdown()
    pbar.close()
    print("Training finished.")

    # 최종 모델 저장
    final_model_path = os.path.join(save_dir, "policy.pth")
    torch.save(actor_value_module.state_dict(), final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script for DAA-CPS")
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        default="problems/cross/cross_1.json",
        help="Path to the problem file",
    )
    parser.add_argument(
        "--total_frames", type=int, default=1_000_000, help="Total frames to train for"
    )
    parser.add_argument(
        "--frames_per_batch",
        type=int,
        default=1024,
        help="Frames collected per data collection phase",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=64,
        help="Mini-batch size for training updates",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train on each batch of data",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient for loss")
    parser.add_argument("--save_dir", type=str, default="artifacts", help="루트 저장 폴더")
    
    # [추가] 시드 옵션
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args)