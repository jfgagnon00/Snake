import numpy as np

from snake.core import RandomProxy
from snake.game import GameAction
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.StateProcessor import _StateProcessor

class AgentRandom(AgentBase):
    """
    Agent qui prend une action aléatoire.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._numGameActions = len(GameAction)
        self._gameActions = list(GameAction)
        self._stateProcessor = _StateProcessor()

    def getAction(self, state, *args):
        _, _, actionFlags = self._stateProcessor(state)

        actionFlags = np.array(actionFlags, dtype=np.float32)
        actionsAvailable = np.nonzero(actionFlags)[0]

        sum_ = 0
        if len(actionsAvailable) > 0:
            lastActionProbs = actionsAvailable.copy()
            sum_ = lastActionProbs.sum()

        if sum_ < 1e-6:
            actionsAvailable = np.arange(self._numGameActions)
            lastActionProbs = np.ones(self._numGameActions, dtype=np.float32)
            sum_ = lastActionProbs.sum()

        lastActionProbs /= sum_

        intAction = RandomProxy.choice(actionsAvailable, p=lastActionProbs)
        return self._gameActions[intAction]
