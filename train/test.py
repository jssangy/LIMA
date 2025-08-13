# test_policy_inspect.py
import torch
import numpy as np
from torchrl.envs.libs.gym import GymWrapper
from tensordict.nn import TensorDictModule
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import OneHotCategorical
from torchrl.envs import ExplorationType, set_exploration_type
from tensordict import TensorDict

from train.gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_rl_action(controller, deterministic=True, debug=True):
    """RL 정책을 사용해 액션을 생성하고 상세 정보를 출력합니다."""
    if not controller.use_rl or not controller.rl_policy:
        return [0, 0, 0, 0, 0]
    
    observation = controller.get_observation_for_rl()
    if observation is None:
        return [0, 0, 0, 0, 0]
    
    if debug:
        print(f"RL State: {observation}")
    
    # 모델이 있는 디바이스 확인
    device = next(controller.rl_policy.parameters()).device
    
    with torch.no_grad():
        # 탐험/착취 모드 설정
        if deterministic:
            controller.rl_policy.eval()
            set_exploration_type(ExplorationType.DETERMINISTIC)
        else:
            controller.rl_policy.eval()
            set_exploration_type(ExplorationType.RANDOM)
        
        # 관찰값을 텐서로 변환
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        td_input = TensorDict({"observation": obs_tensor}, batch_size=[1])
        
        # 모델 실행
        td_output = controller.rl_policy(td_input)
        
        # 액션 추출
        if "action" in td_output.keys():
            action_tensor = td_output["action"].cpu().squeeze()
        else:
            if debug:
                print("RL Action: [0, 0, 0, 0, 0] (no action key)")
            return [0, 0, 0, 0, 0]
        
        # [핵심] logits 추출하여 확률 계산
        if "logits" in td_output.keys():
            logits = td_output["logits"].cpu().squeeze()  # [5, 4] 형태
            
            if debug:
                print(f"\n=== 액션 분석 ===")
                print(f"Logits shape: {logits.shape}")
                print(f"Action tensor shape: {action_tensor.shape}")
                
                # 각 헤드별 확률과 argmax 출력
                action_names = ["North Force", "East Force", "South Force", "West Force", "Center Direction"]
                
                for i, name in enumerate(action_names):
                    head_logits = logits[i]  # i번째 헤드의 logits
                    head_probs = torch.softmax(head_logits, dim=0)  # 확률로 변환
                    head_argmax = torch.argmax(head_logits).item()  # argmax
                    
                    print(f"\n{name} (Head {i}):")
                    print(f"  Logits: {head_logits.numpy()}")
                    print(f"  Probs:  {head_probs.numpy()}")
                    print(f"  Argmax: {head_argmax}")
                    
                    # 선택된 액션 표시
                    if action_tensor.dim() == 2:  # OneHot 형태
                        selected_action = torch.argmax(action_tensor[i]).item()
                    else:  # 이미 인덱스 형태
                        selected_action = action_tensor[i].item()
                    
                    print(f"  Selected: {selected_action}")
                    print(f"  Match: {'✓' if selected_action == head_argmax else '✗'}")
        
        # 액션 변환
        if action_tensor.dim() == 2:  # OneHot 벡터인 경우
            action_indices = torch.argmax(action_tensor, dim=1).tolist()
            if debug:
                print(f"\nFinal Action (OneHot→Index): {action_indices}")
                print(f"OneHot Action Tensor:\n{action_tensor.numpy()}")
        elif action_tensor.dim() == 1:  # 이미 인덱스 형태
            action_indices = action_tensor.tolist()
            if debug:
                print(f"\nFinal Action (Index): {action_indices}")
        else:
            if debug:
                print(f"Unexpected action tensor shape: {action_tensor.shape}")
            return [0, 0, 0, 0, 0]
        
        return action_indices

# 1) Env
env = GymWrapper(GymEnv(prob_path="problems/cross/cross_1.json"), device=DEVICE)

# 2) 훈련 때와 동일하게 ActorValueOperator 구성 → policy 추출
state_dim = env.observation_spec["observation"].shape[-1]
common_net = CommonNet(state_dim).to(DEVICE).eval()
policy_head = PolicyHead().to(DEVICE).eval()
value_head = ValueHead().to(DEVICE).eval()

common_op = TensorDictModule(common_net, in_keys=["observation"], out_keys=["hidden"])
logits_op = TensorDictModule(policy_head, in_keys=["hidden"], out_keys=["logits"])
value_op = TensorDictModule(value_head, in_keys=["hidden"], out_keys=["state_value"])

policy_op = ProbabilisticActor(
    module=logits_op, 
    spec=env.action_spec,
    in_keys=["logits"], 
    out_keys=["action"],
    distribution_class=OneHotCategorical, 
    return_log_prob=False
)

avo = ActorValueOperator(common_op, policy_op, value_op)
policy = avo.get_policy_operator().to(DEVICE).eval()

# 3) 가중치 로드
CKPT = "checkpoint/policy_last.pth"  # 파일 경로 수정하세요
try:
    sd = torch.load(CKPT, map_location=DEVICE)
    policy.load_state_dict(sd, strict=True)
    print(f"✓ Loaded: {CKPT}")
except FileNotFoundError:
    print(f"✗ Checkpoint not found: {CKPT}")
    print("  Using randomly initialized weights for demonstration")

# 4) 더미 컨트롤러 클래스
class DummyController:
    def __init__(self, policy):
        self.use_rl = True
        self.rl_policy = policy
        self._obs = None
    
    def get_observation_for_rl(self):
        return self._obs

ctrl = DummyController(policy)

# 5) 테스트 실행
print(f"State dimension: {state_dim}")
print(f"Action space: {env.action_spec}")

td = env.reset()
ctrl._obs = td["observation"].cpu().numpy()

print("\n" + "="*50)
print("=== STEP 0 (Initial State) ===")
print("="*50)
action_0 = get_rl_action(ctrl, deterministic=True, debug=True)

for t in range(1, 4):
    # 랜덤 액션으로 환경 업데이트
    with torch.no_grad():
        a = env.action_spec.rand().to(DEVICE)
        td = env.step(td.update({"action": a}))
    
    ctrl._obs = td["next", "observation"].cpu().numpy()
    
    print("\n" + "="*50)
    print(f"=== STEP {t} ===")
    print("="*50)
    action_t = get_rl_action(ctrl, deterministic=True, debug=True)

print("\n" + "="*50)
print("=== 테스트 완료 ===")
print("="*50)