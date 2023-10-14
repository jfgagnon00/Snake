import numpy as np

from game import GameAction
from .AgentBase import AgentBase

class AgentRandom(AgentBase):
    """
    Agent qui prend une action aléatoire.
    """
    def __init__(self, trainConfig, simulationConfig):
        super().__init__(trainConfig, simulationConfig)
        self._actions = [
            GameAction.TURN_LEFT,
            GameAction.TURN_RIGHT,
            GameAction.FORWARD
        ]

    def getAction(self, state):
        return np.random.choice(self._actions)
