import numpy as np

from copy import deepcopy
from io import StringIO
from pprint import pprint
from snake.game import GameAction

from .NodeException import _NodeException


class _Node(object):
    def __init__(self, state, info, done, won):
        numActions = len(GameAction)

        self._simulationState = deepcopy(info["simulation_state"])
        self._availableActions = deepcopy(info["available_actions"])
        self.done = done
        self._won = won

        self.visitCount = 0

        assert len(self._availableActions) == numActions

        # 1 element par action
        self.Q = np.zeros(numActions, dtype=np.float32)
        self.N = np.zeros(numActions, dtype=np.float32)
        self.W = np.zeros(numActions, dtype=np.float32)
        self.child = [None] * numActions

    @property
    def simulationState(self):
        return self._simulationState

    @property
    def availableActions(self):
        return self._availableActions

    @property
    def won(self):
        return self._won

    def validate(self, state, info):
        if "simulation_state" not in info or \
           "available_actions" not in info:
            raise _NodeException("Clef manquante dans info", state, info, self)

        # if info["simulation_state"] != self._simulationState or \
        #    not np.array_equal(info["available_actions"], self._availableActions):
        #     raise _NodeException(f"Clef differente", state, info, self)

        availableActions = self.availableActions.sum()
        if (self.done and availableActions != 0) or \
           (not self.done and availableActions == 0):
            raise _NodeException(f"Done inconsistant", state, info, self)

        numCells = self.simulationState.gridWidth * self.simulationState.gridHeight
        snakeLength = self.simulationState.snake.length

        if self.won and not self.done or \
           self.won and snakeLength != numCells:
            raise _NodeException(f"Won inconsistant", state, info, self)

    def __repr__(self):
        with StringIO() as stream:
            pprint(vars(self), stream=stream)
            return stream.getvalue()
