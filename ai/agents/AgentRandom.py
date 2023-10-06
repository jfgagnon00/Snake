import numpy as np

from game import GameAction

class AgentRandom():
    def __init__(self):
        self._actions = [
            GameAction.TURN_LEFT,
            GameAction.TURN_RIGHT,
            GameAction.FORWARD,
            GameAction.COUNT
        ]

    def getAction(self, observation):
        return np.random.choice(self._actions)
