import numpy as np

from copy import deepcopy
from snake.game import GameAction
from .NodeException import _NodeException


class _Node(object):
    def __init__(self, state, info, done):
        self._state = deepcopy(state)
        self._state["snake_bodyparts"] = deepcopy(info["snake_bodyparts"])
        self._done = done

        # 1 element par action
        self._P = None
        self._Q = None
        self._N = None
        self._V = None  # array de numpy array; V de chaque simulation pour chaque action
        self._child = None

        # internal management pour Mcts.getAction()
        self._vistCount = 0

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
                raise _NodeException("Clef manquante dans state", state, info, self._state)

            if self._state[k] != state[k]:
                raise _NodeException(f"Clef '{k}' differente", state, info, self._state)

        infoKeys = ["snake_bodyparts"]
        for k in infoKeys:
            if not k in info:
                raise _NodeException("Clef manquante dans info", state, info, self._state)

            if self._state[k] != info[k]:
                raise _NodeException(f"Clef '{k}' differente", state, info, self._state)

    def ucb(self, cpuct):
        assert not self.isLeaf
        return self._Q + cpuct * self._P * np.sqrt(self._N.sum()) / (1 + self._N)

    def expand(self, nodeFactory, env, modelCallable):
        actions = list(GameAction)
        numActions = len(actions)

        self._Q = np.zeros(numActions, dtype=np.float32)
        self._N = np.zeros(numActions, dtype=np.float32)
        self._V = [np.empty(1, dtype=np.float32) for _ in actions]

        p, v, actionFlags = modelCallable(self._state)
        self._P = p.detach().numpy()

        self._child = []
        for action in actions:
            env.reset(options=self._state)

            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            node = nodeFactory.getOrCreate(state, info, done)

            self._child.append(node)

        return v.detach().value()
