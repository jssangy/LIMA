import gymnasium as gym
from gymnasium import spaces
import numpy as np

from Environment import ENV

class GymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prob_path):
        super().__init__()
        self.env = ENV(prob_path)

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
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()

        obs = self._get_observation().astype(np.float32, copy=False)
        info = {"agv_in_intersection": np.array(self._agv_in_intersection(), dtype=np.int8)}
        return obs, info

    def step(self, action):
        env_info = self.env.step(action)

        for num, agv in self.env.agv_list.items():
            self.env.controller.get_sensing(num, self.env.network.send(agv.sensing()))

        observation = self._get_observation().astype(np.float32, copy=False)
        reward = float(self._calculate_reward(env_info))
        terminated = bool(not env_info)
        truncated = False
        info = {"agv_in_intersection": np.array(self._agv_in_intersection(), dtype=np.int8)}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        state = self.env.intersection.get_state()
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, env_info):
        reward = -0.01
        for event in self.env.intersection.exit_events:
            if event["correct"]:
                reward += 1.0
            else:
                reward -= 0.5
        return reward

    def _agv_in_intersection(self):
        # 교차로가 비었으면 0, 아니면 1
        return 0 if self.env.intersection.is_empty else 1

    def close(self):
        pass
