from enum import StrEnum, auto

class Rewards(StrEnum):
    EAT = auto()
    MOVE = auto()
    TRAPPED = auto()
    WIN = auto()
    TRUNCATED = auto()
    UNKNOWN = auto()

class ConfigEnvironment(object):
    def __init__(self):
        self.renderFps = 0
        self.rewards = {
            Rewards.EAT: 10,
            Rewards.MOVE: 0,
            Rewards.TRAPPED: -10,
            Rewards.WIN: 30,
            Rewards.TRUNCATED: 0,
            Rewards.UNKNOWN: 0,
        }
