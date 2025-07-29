# main.py 또는 테스트 스크립트에서
from IntersectionEnv import IntersectionEnv
from torchrl.envs.utils import check_env_specs

from Controller import Controller  # Controller 클래스를 가져옵니다.
from Intersection import Intersection  # Intersection 클래스를 가져옵니다.

from Environment import ENV

# Controller와 intersection_data를 준비했다고 가정
# controller = Controller(...)
# intersection_data = ...

# 환경 인스턴스 생성
env = IntersectionEnv(intersection_data, controller)

# 1. 환경 명세 확인 (매우 중요!)
# 이 함수가 오류 없이 통과하면 환경이 잘 정의된 것입니다.
check_env_specs(env)

# 2. 환경 사용 예시
td = env.reset()
print("Initial TensorDict:", td)

# 무작위 행동으로 한 스텝 진행
td = env.step(env.rand_action())
print("TensorDict after one step:", td)

env.close()