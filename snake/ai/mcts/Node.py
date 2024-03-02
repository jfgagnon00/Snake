import numpy as np

from copy import deepcopy
from .NodeException import _NodeException


class _Node(object):
    def __init__(self, parent, env, state, info):
        self._parent = parent
        self._child = None

        # numpy arrays
        self._Q = None
        self._N = None

        self._P = None
        self._V = 0

        self._done = False

        self._env = deepcopy(env)
        self._state = deepcopy(state)
        self._state["snake_bodyparts"] = deepcopy(info["snake_bodyparts"])

    @property
    def isLeaf(self):
        return self._child is None

    @property
    def child(self):
        return self._child

    @property
    def done(self):
        return self._done

    @property
    def V(self):
        return self._V

    def ucb(self, cpuct):
        assert not self._Q is None
        return self._Q + cpuct * self._P * np.sqrt(self._N.sum()) / (1 + self._N)

    def validate(self, state, info):
        stateKeys = [
            "occupancy_grid",
            "head_direction",
            "head_position",
            "food_position",
            "score",
        ]
        for k in stateKeys:
            if not k in state:
                raise _MctsNodeException("Clef manquante dans state", state, info, self._state)

            if self._state[k] != state[k]:
                raise _MctsNodeException(f"Clef '{k}' differente", state, info, self._state)

        infoKeys = ["snake_bodyparts"]
        for k in infoKeys:
            if not k in info:
                raise _MctsNodeException("Clef manquante dans info", state, info, self._state)

            if self._state[k] != info[k]:
                raise _MctsNodeException(f"Clef '{k}' differente", state, info, self._state)

    def expand(self, model, stateProcessingCallable):
        self._Q = np.zeros(self._env.action_space.n, dtype=np.float32)
        self._N = np.zeros(self._env.action_space.n, dtype=np.int32)

        x0, x1, flags = stateProcessingCallable(self._state)
        model(self._state)

        self._P = None
        self._V = None

        self._done = False
