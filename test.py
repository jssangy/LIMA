import torch
import argparse
from tqdm import tqdm

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

# 직접 만든 환경 및 분리된 모델 임포트
# 이 환경들은 이제 단일 교차로를 제어하는 것을 전제로 합니다.
from gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead

from tensordict import TensorDict
from torchrl.data import Unbounded, MultiOneHot   # ✨ 새 alias

BATCH = 8

# 1) Spec 정의
obs_spec = Unbounded(shape=(28,), dtype=torch.float32)
act_spec = MultiOneHot(nvec=[4, 4, 4, 4, 4])     # (5,4) one-hot

# 2) 더미 데이터 생성
obs      = torch.randn(BATCH, *obs_spec.shape)            # (8, 28)
hidden   = torch.randn(BATCH, 128)                        # (8, 128)
logits   = torch.randn(BATCH, 5, 4)                       # (8, 5, 4)

# ‼ rand() 호출 시 튜플 · torch.Size로 배치 shape 전달
action   = act_spec.rand((BATCH,))                        # (8, 5, 4)
log_prob = torch.randn(BATCH, 5)

# 3) TensorDict 구성
td = TensorDict(
    {
        "observation":  obs,
        "hidden":       hidden,
        "logits":       logits,
        "action":       action,
        "log_prob":     log_prob,
        "advantage":    torch.randn(BATCH, 1),
        "value_target": torch.randn(BATCH, 1),
        "reward":       torch.zeros(BATCH, 1),
        "done":         torch.zeros(BATCH, 1, dtype=torch.bool),
    },
    batch_size=[BATCH],
).to("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # --- 1. 설정 및 초기화 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 2. 환경 생성 ---
    # GymEnv는 단일 교차로의 상태를 관측하고, 단일 교차로에 대한 행동을 받습니다.
    base_env = GymEnv(prob_path=args.problem)
    env = GymWrapper(base_env, device=DEVICE)

    # --- 3. 모델, 정책, 가치 함수 정의 ---
    # 관측값(observation)은 단일 교차로의 상태 벡터입니다.
    state_dim = env.observation_spec["observation"].shape[-1]

    # 3.1 각 모듈을 생성하고 TensorDictModule으로 래핑합니다.
    # CommonNet은 교차로의 상태를 입력받아 공통 특징 벡터(hidden)를 출력합니다.
    # Input: (B, 28) -> Output: (B, 128)
    common_net = CommonNet(state_dim).to(DEVICE)
    common_operator = TensorDictModule(
        module=common_net, in_keys=["observation"], out_keys=["hidden"]
    )

    # PolicyHead는 특징 벡터를 입력받아 교차로 제어를 위한 행동의 로짓(logits)을 출력합니다.
    # Input: (B, 128) -> Output: (B, 5, 4)
    policy_net = PolicyHead().to(DEVICE)
    policy_logits = TensorDictModule(
        module=policy_net, in_keys=["hidden"], out_keys=["logits"]
    )

    # ProbabilisticActor는 로짓을 기반으로 실제 행동(action)을 샘플링합니다.
    # env.action_spec은 GymEnv의 MultiDiscrete 공간에 맞춰 자동으로 생성됩니다.
    policy_operator = ProbabilisticActor(
        module=policy_logits,
        spec=env.action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        log_prob_key="log_prob",
    )

    # ValueHead는 특징 벡터를 입력받아 현재 상태의 가치(state_value)를 출력합니다.
    value_net = ValueHead().to(DEVICE)
    value_operator = TensorDictModule(
        module=value_net, in_keys=["hidden"], out_keys=["state_value"]
    )

    # 3.2 세 개의 래핑된 모듈을 ActorValueOperator로 조립합니다.
    # 이는 observation -> hidden -> (logits, state_value) -> (action, state_value)의 전체 흐름을 담당합니다.
    actor_value_module = ActorValueOperator(
        common_operator=common_operator,
        policy_operator=policy_operator,
        value_operator=value_operator,
    )

    # 훈련에 사용할 전체 정책 및 가치 평가 모듈을 가져옵니다.
    policy = actor_value_module.get_policy_operator()
    value_module = actor_value_module.get_value_operator()


    # --- 4. 데이터 수집기 및 리플레이 버퍼 설정 ---
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
    )

    # --- 5. 손실 함수 및 옵티마이저 설정 ---
    advantage_module = GAE(
        gamma=args.gamma,
        lmbda=args.lmbda,
        value_network=value_module,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        loss_critic_type="l2",
    )
    
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script for DAA-CPS")
    parser.add_argument('--problem', '-p', type=str, default='problems/cross/cross_1.json', help='Path to the problem file')
    parser.add_argument("--total_frames", type=int, default=500_000, help="Total frames to train for")
    parser.add_argument("--frames_per_batch", type=int, default=128, help="Frames collected per data collection phase")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch size for training updates")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train on each batch of data")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient for loss")
    
    args = parser.parse_args()
    main(args)