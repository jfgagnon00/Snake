from enum import IntEnum

class Rewards(IntEnum):
    EAT = 0
    MOVE = 1
    TRAPPED = 2
    WIN = 3
    TRUNCATED = 4
    UNKNOWN = 5

class ConfigEnvironment(object):
    def __init__(self):
        self.renderFps = 0
        self.rewards = {
            Rewards.EAT.name: 10,
            Rewards.MOVE.name: 0,
            Rewards.TRAPPED.name: -10,
            Rewards.WIN.name: 30,
            Rewards.TRUNCATED.name: 0,
            Rewards.UNKNOWN.name: 0,
        }
