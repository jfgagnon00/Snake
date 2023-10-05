from gym import Env

class OpenAIGymAdapter(Env):
    def __init__(self):
        super().__init__()

        self.action_space = None
        self.observation_space = None
        self.reward_range = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def close(self):
        pass