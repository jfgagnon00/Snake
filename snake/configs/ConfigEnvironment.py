from enum import StrEnum, auto

class Rewards(StrEnum):
    MOVE = auto()
    EAT = auto()
    COLLISION = auto()
    OUT_OF_BOUNDS = auto()
    WIN = auto()

class ConfigEnvironment():
    def __init__(self):
        self.renderFps = 0
        self.rewards = {
            Rewards.MOVE: 0,
            Rewards.EAT: 10,
            Rewards.COLLISION: -10,
            Rewards.OUT_OF_BOUNDS: -10,
            Rewards.WIN: 30,
        }
