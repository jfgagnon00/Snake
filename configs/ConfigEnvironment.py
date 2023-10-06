class ConfigEnvironment():
    def __init__(self):
        self.renderFps = 0
        self.rewards = {
            "move": 0,
            "eat": 10,
            "collide": -10,
        }