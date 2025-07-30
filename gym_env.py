import gymnasium as gym
from gymnasium import spaces
import numpy as np

from Environment import ENV

class GymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prob_path):
        super().__init__()
        self.env = ENV(prob_path) # 내부적으로 기존 환경 소유

        # 관측 공간(Observation Space) 정의
        # 단일 교차로의 상태 벡터 크기에 맞춰 shape를 설정해야 합니다.
        # 현재 코드는 28로 가정합니다.
        low = []
        high = []
        for _ in range(4):
            low.extend([0] * 6)
            high.extend([1, 1, 1, 1, 1000, 1])
        low.extend([0] * 4)
        high.extend([1] * 4)

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            shape=(28,),
            dtype=np.float32
        )

        # 행동 공간(Action Space) 정의: 5개의 요소로 구성된 단일 벡터
        # 각 요소는 4단계(0,1,2,3)를 가집니다.
        # [반발장N, 반발장E, 반발장S, 반발장W, 중심제어]
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        
        # 초기 상태 가져오기
        obs = self._get_observation()
        info = {}
        
        return obs, info

    def step(self, action):
        # action은 [r0, r1, r2, r3, c] 형태의 numpy 배열입니다.
        # 단일 교차로이므로, action을 바로 내부 환경의 step 함수로 전달합니다.
        env_info = self.env.step(action[0])

        # 결과 계산
        observation = self._get_observation()
        reward = self._calculate_reward(env_info)
        terminated = not env_info
        truncated = False # 시간 초과 등의 이유로 종료될 경우 True로 설정
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """단일 교차로의 현재 상태를 numpy 배열로 반환"""
        # self.env에 있는 단일 intersection 객체의 상태를 직접 가져옵니다.
        state = self.env.intersection.get_state()
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, env_info):
        """내부 환경에서 계산된 보상 값을 사용"""
        # 예: env_info['total_reward'] 값을 그대로 사용하거나 가공
        # 현재는 충돌 등에 대한 간단한 패널티만 부여
        reward = -0.01
        return reward

    def close(self):
        # 필요 시 환경 자원 해제 로직 추가
        pass