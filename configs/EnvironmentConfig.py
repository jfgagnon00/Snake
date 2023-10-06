class EnvironmentConfig():
    MOVE = "move"
    EAT = "eat"
    COLLIDE = "collide"


    def __init__(self):
        self.rewards = {
            EnvironmentConfig.MOVE: 0,
            EnvironmentConfig.EAT: 10,
            EnvironmentConfig.COLLIDE: -10,
        }