import numpy as np

from copy import deepcopy
from .NodeException import _NodeException


class _Node(object):
    def __init__(self, state, info, done, won):
        self._state = {}

        # dupliquer les clef necessaire seulement
        for k in _Node.stateKeys():
            self._state[k] = deepcopy(state[k])

        for k in _Node.infoKeys():
            self._state[k] = deepcopy(info[k])

        self._done = done
        self._won = won

        # 1 element par action
        self.P = None
        self.Q = None
        self.N = None
        self.W = None  # somme de tous les V pour chaque action
        self.child = None

        # internal management pour Mcts.getAction()
        self.vistCount = 0

    @staticmethod
    def stateKeys():
        return [
            "occupancy_grid",
            "head_direction",
            "head_position",
            "food_position",
            "score",
            "available_actions",
        ]

    @staticmethod
    def infoKeys():
        return ["snake_bodyparts"]

    @property
    def isLeaf(self):
        return self.child is None

    @property
    def state(self):
        return self._state

    @property
    def done(self):
        return self._done

    @property
    def won(self):
        return self._won

    def validate(self, state, info):
        for k in _Node.stateKeys():
            if not k in state:
                raise _NodeException("Clef manquante dans state", state, info, self._state)

            if k == "score":
                equal_ = self._state[k] == state[k]
            else:
                equal_ = np.array_equal(self._state[k], state[k])

            if not equal_:
                raise _NodeException(f"Clef '{k}' differente", state, info, self._state)

        for k in _Node.infoKeys():
            if not k in info:
                raise _NodeException("Clef manquante dans info", state, info, self._state)

            if not np.array_equal(self._state[k], info[k]):
                raise _NodeException(f"Clef '{k}' differente", state, info, self._state)
