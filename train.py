import os
import time
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import default_info_dict_reader
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule, TensorDictSequential

from module.gym_env import GymEnv
from module.model import GNNBody, Actor, Critic


def set_seed(seed):
    """모든 랜덤 시드를 고정하여 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 추가 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ All random seeds set to: {seed}")


def main(args):
    # --- 0. 시드 설정 (가장 먼저) ---
    if args.seed is not None: set_seed(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ... wandb 및 저장 경로 설정 ...
    exp_name = f"lr{args.lr:.0e}_gm{args.gamma}_lam{args.lmbda}_clip{args.clip_epsilon}_ent{args.entropy_coeff:.0e}"
    wandb.init(project="GNN_MAPPO", config=vars(args), name=exp_name)
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. 환경 생성 ---
    base_env = GymEnv(prob_path=args.problem)

    # 멀티 에이전트 설정을 위한 그룹 맵 생성
    agent_ids = base_env.agent_ids
    n_agents = len(agent_ids)
    group_map = {"agents": agent_ids}
    env = GymWrapper(base_env, device=DEVICE, group_map=group_map,spec_keys=["observation", "action", "reward", "done"])
    

    # --- 2. GNN 기반 액터-크리틱 모델 설정 ---    
    # 공유 GNN Body
    gnn_body = TensorDictModule(
        module=GNNBody(input_dim=24, hidden_dim=128),
        in_keys=[("agents", "observation", "nodes"), 
                 ("agents", "observation", "edge_index"),
                 ("agents", "observation", "node_mask")], # 마스크도 GNN에 전달될 수 있음 (모델 내부에서 사용 안해도)
        out_keys=[("agents", "node_embeddings")] # [B, N_agents, N_max, hidden_dim]
    )

    # 독립적인 Actor 헤드
    # GNN Body의 출력을 받아 각 노드에 대해 독립적으로 행동 로짓 계산
    policy_net = TensorDictModule(
        module=Actor(hidden_dim=128, action_dims=5, action_levels=4),
        in_keys=[("agents", "node_embeddings")],
        out_keys=[("agents", "logits")] # [B, N_agents, N_max, 5, 4]
    )
    
    # ProbabilisticActor는 각 에이전트의 MultiDiscrete 행동을 처리
    policy_operator = ProbabilisticActor(
        module=policy_net,
        spec=env.action_spec["agents"],
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=OneHotCategorical, # MultiDiscrete에 적합
        return_log_prob=True,
    )

    # 중앙화된 Critic 헤드
    # Critic은 GNN Body의 출력을 받아 서브그래프 전체의 가치를 평가
    # Critic 모델은 global_mean_pool을 사용하므로, 배치 정보를 전달해야 함
    # torchrl이 내부적으로 처리하므로 키 매핑만 정확히 해주면 됨
    value_net = TensorDictModule(
        module=Critic(hidden_dim=128),
        # Critic은 모든 노드 임베딩을 입력으로 받음
        in_keys=[("agents", "node_embeddings"), ("agents", "observation", "node_mask")],
        out_keys=["state_value"] # 중앙 Critic은 단일 가치 출력
    )
    
    # 최종 Actor-Value 모듈 통합
    actor_value_module = ActorValueOperator(
        common_operator=gnn_body,
        policy_operator=policy_operator,
        value_operator=value_net,
    ).to(DEVICE)

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
        gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        loss_critic_type="l2",
        reward_key=("next", "agents", "reward")
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # --- 5. 훈련 루프 ---
    pbar = tqdm(total=args.total_frames)
    total_collected_frames = 0

    for i, tensordict_data in enumerate(collector):
        total_collected_frames += tensordict_data.numel()
        pbar.update(tensordict_data.numel())

        # GAE 계산
        with torch.no_grad():
            advantage_module(tensordict_data)

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

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

        # 로깅
        avg_reward = tensordict_data.get(("next", "agents", "reward")).mean().item()
        wandb.log({
            "reward/mean": avg_reward,
            "loss/objective": loss_vals["loss_objective"].item(),
            "loss/critic": loss_vals["loss_critic"].item(),
            "loss/entropy": loss_vals["loss_entropy"].item(),
        }, step=total_collected_frames)
        pbar.set_description(f"Iter {i+1}, Reward: {avg_reward:.4f}")

    collector.shutdown()
    pbar.close()
    print("Training finished.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPPO-GNN Training Script for DAA-CPS")
    parser.add_argument("--problem", type=str, default="problems/cross/cross_1.json")
    parser.add_argument("--total_frames", type=int, default=1_000_000)
    parser.add_argument("--frames_per_batch", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--save_dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)