import torch
import torch.optim as optim
import torchrl
from torchrl.envs import ParallelEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator

# --- 핵심 변경: 필요한 모든 모듈 임포트 ---
from Environment import ENV # 메인 시뮬레이션 환경
from model import ActorCritic

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # --- 1. 하이퍼파라미터 및 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # PPO 하이퍼파라미터 (필요에 맞게 조정)
    TOTAL_FRAMES = 1_000_000
    FRAMES_PER_BATCH = 2048
    MINI_BATCH_SIZE = 256
    NUM_EPOCHS = 10
    LR = 3e-4
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPSILON = 0.2

    # --- 2. 시뮬레이션 및 RL 환경 초기화 ---
    print("Loading main simulation environment...")
    main_simulation = ENV("problems/cross/cross_1.json") 

    controller = main_simulation.controller
    controller.use_rl = True

    num_intersections = len(controller.intersections)
    create_env_fn_list = [
        lambda i=i: controller.intersections[i] for i in range(num_intersections)
    ]
    env = ParallelEnv(num_workers=num_intersections, create_env_fn=create_env_fn_list)

    # --- 3. PPO 에이전트 구성 ---
    base_model = ActorCritic(state_dim=29, action_dim=3)
    actor_critic_model = TensorDictModule(
        module=base_model,
        in_keys=["observation"],
        out_keys=["logits", "state_value"]
    ).to(DEVICE)
    actor = ProbabilisticActor(
        module=actor_critic_model,
        in_keys=["logits"],
        out_keys=["action", "log_prob"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    ).to(DEVICE)
    critic = ValueOperator(
        module=actor_critic_model,
        in_keys=["observation"],
    ).to(DEVICE)
    advantage_module = GAE(gamma=GAMMA, lmbda=LAMBDA, value_network=critic)
    advantage_module.set_keys(value="state_value")
    loss_module = ClipPPOLoss(actor=actor, critic=critic, clip_epsilon=CLIP_EPSILON, entropy_coef=0.01)
    optimizer = optim.Adam(loss_module.parameters(), lr=LR)

    # --- 4. 데이터 수집기 및 리플레이 버퍼 ---
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor,
        total_frames=TOTAL_FRAMES,
        frames_per_batch=FRAMES_PER_BATCH,
        device=DEVICE,
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=FRAMES_PER_BATCH),
        batch_size=MINI_BATCH_SIZE,
    )

    # --- 5. 훈련 루프 ---
    print("Training starts...")
    for i, data in enumerate(collector):
        main_simulation.update()
        
        with torch.no_grad():
            advantage_module(data)

        replay_buffer.extend(data)

        for _ in range(NUM_EPOCHS):
            for batch in replay_buffer:
                loss_dict = loss_module(batch)
                loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if i % 10 == 0:
            print(f"Step {i*FRAMES_PER_BATCH}: Total Loss = {loss.item():.4f}, Mean Reward = {data['next']['reward'].mean().item():.4f}")

    print("Training finished.")
    collector.shutdown()