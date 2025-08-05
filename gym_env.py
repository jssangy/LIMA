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
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4, 4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        
        obs = self._get_observation()
        info = {}
        
        return obs, info

    def step(self, action):
        print(f"Action taken: {action}")
        env_info = self.env.step(action[0])

        observation = self._get_observation()
        reward = self._calculate_reward(env_info)
        terminated = not env_info
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        state = self.env.intersection.get_state()
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, env_info):
        reward = -0.01
        return reward

    def close(self):
        pass