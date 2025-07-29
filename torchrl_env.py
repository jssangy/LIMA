import torch
from torchrl.envs import EnvBase
from torchrl.data import TensorDict, BoundedTensorSpec, UnboundedContinuousTensorSpec

# 기존에 작성한 Intersection 클래스와 Controller를 가져옵니다.
from Intersection import Intersection
# from Controller import Controller # Controller도 필요합니다.

class IntersectionEnv(EnvBase):
    """
    Intersection 시뮬레이션을 위한 Custom TorchRL 환경.
    """
    def __init__(self, intersection_data, controller_ref, device="cpu"):
        """
        환경을 초기화하고, 상태/행동/보상 명세를 정의합니다.
        """
        # 1. EnvBase의 __init__ 호출
        super().__init__(device=device)
        
        # 2. 기존 시뮬레이션 로직을 내부 객체로 가짐
        self.intersection = Intersection(intersection_data, controller_ref)
        
        # 3. 명세(Specs) 정의
        # 상태: 29차원의 연속적인 값
        self.observation_spec = UnboundedContinuousTensorSpec(
            shape=(29,), 
            device=self.device
        )
        # 행동: 2개의 이산적인 행동 (예: 0=남북통행, 1=동서통행)
        self.action_spec = BoundedTensorSpec(
            low=0, 
            high=1, # 행동의 최대값 (행동 개수 - 1)
            shape=(1,), 
            dtype=torch.int64, 
            device=self.device
        )
        # 보상: 1차원의 연속적인 값
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), 
            device=self.device
        )

    def _reset(self, tensordict=None):
        """
        환경을 리셋하고 초기 상태를 반환합니다.
        """
        # 실제 리셋 로직 (예: Controller의 AGV 위치, 작업 등을 초기화)
        # self.intersection.controller.reset() 
        
        # Intersection 클래스에서 초기 상태를 가져옴
        initial_state = self.intersection.get_state()
        
        # 결과를 TensorDict 형태로 포장하여 반환
        return TensorDict({
            "observation": torch.tensor(initial_state, dtype=torch.float32, device=self.device)
        }, batch_size=[], device=self.device)

    def _step(self, tensordict):
        """
        주어진 행동을 수행하고 다음 상태, 보상, 종료 여부를 반환합니다.
        """
        # 입력 tensordict에서 행동을 추출
        action = tensordict["action"].item()
        
        # Intersection 클래스의 step 함수를 호출 (이 함수는 직접 구현해야 함)
        # next_state, reward, done = self.intersection.step(action)
        
        # --- 임시 step 함수 (실제 구현 필요) ---
        # 예시: 행동에 따라 AGV를 움직이고, 보상을 계산하는 로직
        # self.intersection.controller.update_traffic_light(self.intersection.id, action)
        # self.intersection.controller.move_agvs()
        # reward = self.intersection.calculate_reward()
        # next_state = self.intersection.get_state()
        # done = False # 에피소드 종료 조건
        
        # 아래는 임시 값입니다. 실제 값으로 교체해야 합니다.
        next_state = self.intersection.get_state()
        reward = 0.0 
        done = False
        # --- 임시 코드 끝 ---

        # 결과를 TensorDict 형태로 포장하여 반환
        out = TensorDict({
            "observation": torch.tensor(next_state, dtype=torch.float32, device=self.device),
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device),
        }, batch_size=[], device=self.device)
        return out

    def _set_seed(self, seed):
        # 환경의 무작위성을 제어하기 위한 시드 설정 (필요시 구현)
        pass