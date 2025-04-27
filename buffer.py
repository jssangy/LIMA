class MultiAgentBuffer:
    def __init__(self):
        self.state = []
        self.actions = []
        self.rewards = []
        self.next_state = []

    def store(self, state, actions, rewards, next_state):
        self.state.append(state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_state.append(next_state)

    def get_all(self):
        return self.state, self.actions, self.rewards, self.next_state

    def clear(self):
        self.state = []
        self.actions = []
        self.rewards = []
        self.next_state = []