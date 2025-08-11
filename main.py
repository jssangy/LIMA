from Environment import ENV
from GUI import GUI
import argparse
import torch
from model import CommonNet, PolicyHead, ValueHead
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.modules.distributions import OneHotCategorical
from tensordict.nn import TensorDictModule

def load_ppo_policy(state_dim, policy_path='ppo_policy.pth'):
    """훈련된 PPO 정책 모델을 불러옵니다."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # train.py와 동일하게 모델 구조를 정의합니다.
    common_net = CommonNet(state_dim).to(DEVICE)
    common_operator = TensorDictModule(module=common_net, in_keys=["observation"], out_keys=["hidden"])
    
    policy_net = PolicyHead().to(DEVICE)
    policy_logits = TensorDictModule(module=policy_net, in_keys=["hidden"], out_keys=["logits"])
    
    # 정책 연산자 정의 (train.py와 동일)
    policy_operator = ProbabilisticActor(
        module=policy_logits,
        spec=None,  # 실행 시에는 spec이 필요 없습니다
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=False,
    )
    
    # ValueHead는 정책 실행에는 필요 없지만, ActorValueOperator 구조를 맞추기 위해 생성
    value_net = ValueHead().to(DEVICE)
    value_operator = TensorDictModule(module=value_net, in_keys=["hidden"], out_keys=["state_value"])
    
    # train.py와 동일한 ActorValueOperator 구조 생성
    actor_value_module = ActorValueOperator(
        common_operator=common_operator,
        policy_operator=policy_operator,
        value_operator=value_operator,
    )
    
    # 정책 부분만 추출
    policy_module = actor_value_module.get_policy_operator()
    
    try:
        # 훈련된 가중치 불러오기
        policy_state_dict = torch.load(policy_path, map_location=DEVICE)
        policy_module.load_state_dict(policy_state_dict)
        print(f"PPO policy loaded from {policy_path}")
        return policy_module
    except FileNotFoundError:
        print(f"Policy file {policy_path} not found. Running without PPO model.")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem', '-p',
        type=str,
        default='problems/cross/cross_1.json',
        help='Problem json file path (e.g. problems/cross/cross_1.json)'
    )
    parser.add_argument(
        '--policy_path',
        type=str,
        default='checkpoint/policy_best.pth',
        help='Path to the trained PPO policy model file'
    )
    args = parser.parse_args()

    # 환경 생성
    env = ENV(args.problem)
    
    # PPO 모델 로딩 (28차원은 상태 벡터 크기)
    policy = load_ppo_policy(state_dim=28, policy_path=args.policy_path)
    if policy:
        env.set_rl_policy(policy)
    
    # GUI 실행
    gui = GUI(env)

if __name__ == "__main__":
    main()