from game import GameAction
from random import choice
from .AgentBase import AgentBase

class AgentRandom(AgentBase):
    """
    Agent qui prend une action aléatoire.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._actions = list(GameAction)

    def getAction(self, *args):
        action = choice(self._actions)
        return GameAction(action)
