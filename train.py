import os
import time
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
import wandb

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import default_info_dict_reader
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

from gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead


def main(args):
    # --- 1. 설정 및 초기화 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="MAPF",
        config=vars(args),
        name=args.exp_name or f"ppo_run_{wandb.util.generate_id()}",
    )

    # 저장 경로 결정
    run_name = args.exp_name or (wandb.run.name if wandb.run is not None else f"run_{int(time.time())}")
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 환경 생성 ---
    base_env = GymEnv(prob_path=args.problem)
    env = GymWrapper(base_env, device=DEVICE)
    env.set_info_dict_reader(default_info_dict_reader(keys=["agv_in_intersection"]))

    # --- 3. 액터-크리틱 모델 설정 ---
    state_dim = env.observation_spec["observation"].shape[-1]
    common_net = CommonNet(state_dim).to(DEVICE)
    common_operator = TensorDictModule(
        module=common_net, in_keys=["observation"], out_keys=["hidden"]
    )
    policy_net = PolicyHead().to(DEVICE)
    policy_logits = TensorDictModule(
        module=policy_net, in_keys=["hidden"], out_keys=["logits"]
    )
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
    value_operator = TensorDictModule(
        module=value_net, in_keys=["hidden"], out_keys=["state_value"]
    )
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
        storing_device=DEVICE,  # 메모리 아끼려면 "cpu"로 변경 가능
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=DEVICE),
        batch_size=args.mini_batch_size,
        sampler=SamplerWithoutReplacement(),
    )

    # --- 5. 손실 함수 및 옵티마이저 설정 ---
    advantage_module = GAE(
        gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        loss_critic_type="l2",
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # 체크포인트 유틸
    def save_ckpt(tag: str, frames: int):
        tmp_pol = os.path.join(save_dir, f"policy_{tag}.pth.tmp")
        tmp_val = os.path.join(save_dir, f"value_{tag}.pth.tmp")
        torch.save(policy.state_dict(), tmp_pol)
        torch.save(value_module.state_dict(), tmp_val)
        os.replace(tmp_pol, os.path.join(save_dir, f"policy_{tag}.pth"))
        os.replace(tmp_val, os.path.join(save_dir, f"value_{tag}.pth"))
        state = {
            "config": vars(args),
            "policy": policy.state_dict(),
            "value": value_module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frames": frames,
            "timestamp": time.time(),
        }
        tmp_all = os.path.join(save_dir, f"trainstate_{tag}.pt.tmp")
        torch.save(state, tmp_all)
        os.replace(tmp_all, os.path.join(save_dir, f"trainstate_{tag}.pt"))

    # --- 6. 훈련 루프 ---
    logs = defaultdict(list)
    pbar = tqdm(total=args.total_frames)
    total_collected_frames = 0
    best_metric = float("-inf")

    for i, tensordict_data in enumerate(collector):
        total_collected_frames += tensordict_data.numel()

        # (A) 전체 롤아웃에 대해 GAE/타깃 1회 계산 (마스크 이전!)
        with torch.no_grad():
            advantage_module(tensordict_data)

        # (B) PPO 배치 동안 old 값 고정
        for k in ["advantage", "value_target", "sample_log_prob"]:
            if k in tensordict_data.keys(True, True):
                tensordict_data.set_(k, tensordict_data.get(k).detach())

        # (C) 현재 상태 기준 게이트 마스크 (교차로일 때만 Actor 업데이트)
        gate_mask = tensordict_data.get("agv_in_intersection").bool()
        tensordict_data.set("gate_mask", gate_mask)
        # 선택: 게이트 비율 로깅
        if gate_mask.numel() > 0:
            wandb.log({"gate/active_ratio": gate_mask.float().mean().item()}, step=total_collected_frames)

        # --- 훈련 로직 ---
        total_loss_objective = 0.0
        total_loss_critic = 0.0
        total_loss_entropy = 0.0
        update_count = 0

        for _ in range(args.num_epochs):
            # 배치 평탄화 → 버퍼에 넣고, 미니배치마다 게이팅
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            for sub_data in replay_buffer:
                # (1) 정책/엔트로피: 교차로 샘플만
                sub_mask = sub_data.get("gate_mask")
                if sub_mask.any():
                    sub_actor = sub_data[sub_mask]
                    loss_vals_actor = loss_module(sub_actor)
                    loss_pi = (
                        loss_vals_actor["loss_objective"]
                        + loss_vals_actor["loss_entropy"]
                    )
                else:
                    loss_pi = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                    loss_vals_actor = {
                        "loss_objective": torch.tensor(0.0, device=DEVICE),
                        "loss_entropy": torch.tensor(0.0, device=DEVICE),
                    }

                # (2) 가치함수: 전체 샘플
                loss_vals_critic = loss_module(sub_data)
                loss_v = loss_vals_critic["loss_critic"]

                # (3) 합산 후 한 번만 스텝
                loss = loss_pi + loss_v
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # 로깅 누적
                total_loss_objective += float(loss_vals_actor["loss_objective"])
                total_loss_entropy += float(loss_vals_actor["loss_entropy"])
                total_loss_critic += float(loss_v)
                update_count += 1

            replay_buffer.empty()

        # --- 로깅 ---
        pbar.update(tensordict_data.numel())
        avg_reward = tensordict_data["next", "reward"].mean().item()
        logs["reward"].append(avg_reward)
        pbar.set_description(f"Iter {i+1}, Reward: {avg_reward:.4f}")

        if update_count > 0:
            wandb.log(
                {
                    "reward": avg_reward,
                    "loss/objective": total_loss_objective / update_count,
                    "loss/critic": total_loss_critic / update_count,
                    "loss/entropy": total_loss_entropy / update_count,
                },
                step=total_collected_frames,
            )

        # 주기 저장
        if args.save_every > 0 and (total_collected_frames % args.save_every == 0):
            save_ckpt(f"step{total_collected_frames}", total_collected_frames)

        # best 저장 (현재는 avg_reward 기준 — 필요시 throughput 기준으로 교체)
        if avg_reward > best_metric:
            best_metric = avg_reward
            save_ckpt("best", total_collected_frames)

    collector.shutdown()
    pbar.close()
    print("Training finished.")

    # 마지막 저장
    save_ckpt("last", total_collected_frames)
    print(f"Saved checkpoints to {save_dir}")

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
    parser.add_argument(
        "--entropy_coeff", type=float, default=0.01, help="Entropy coefficient for loss"
    )
    # 체크포인트 옵션
    parser.add_argument("--save_dir", type=str, default="artifacts", help="루트 저장 폴더")
    parser.add_argument("--exp_name", type=str, default=None, help="저장/로그용 런 이름(없으면 wandb.run.name)")
    parser.add_argument("--save_every", type=int, default=0, help="N 프레임마다 주기 저장(0이면 비활성)")

    args = parser.parse_args()
    main(args)
