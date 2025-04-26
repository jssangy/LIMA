class MultiAgentBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []

    def store(self, obs, actions, rewards, next_obs):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_obs.append(next_obs)

    def get_all(self):
        return self.obs, self.actions, self.rewards, self.next_obs

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []