from snake.core import RandomProxy
from snake.game import GameAction
from snake.ai.agents.AgentBase import AgentBase

class AgentRandom(AgentBase):
    """
    Agent qui prend une action aléatoire.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._actions = list(GameAction)
        self._actionsIndex = range(len(self._actions))

    def getAction(self, *args):
        action = RandomProxy.choice(self._actionsIndex)
        return GameAction(self._actions[action])
