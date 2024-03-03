import numpy as np

from snake.game import GameAction

from .NodeFactory import _NodeFactory


class _Mcts(object):
    def __init__(self, model, trainConfig):
        self._model = model
        self._actions = list(GameAction)
        self._numActions = len(GameAction)

        self._cpuct = trainConfig.mcts.cpuct
        self._numIterations = trainConfig.mcts.numIterations
        self._maxVisitCount = trainConfig.maxVisitCount

        self.reset()

    def initEnv(self, env):
        self._env = env

    def reset(self):
        self._nodeFactory = _NodeFactory()

    def search(self, root, state, info):
        if root is None:
            root = self._nodeFactory.getOrCreate(state, info, False)

        root.validate(state, info)

        # while root.N.sum() < 100:
        #     node = select(root)

        #     if node.done:
        #         # propager valeur
        #         pass
        #     else:
        #         node.expand()

        #     backpropagation(node)

        # newNode, intAction, newPolicy, value = None, 0, None, None

        # return newNode, \
        #     intAction, \
        #     newPolicy, \
        #     value
        pass

    def _select(self, node):
        while not node.isLeaf:
            ucb = node.ucb()
            index = np.argmax(ucb)
            node = node.child[index]

        return node

    def _expand(self, node):
        node.Q = np.zeros(self._numActions, dtype=np.float32)
        node.N = np.zeros(self._numActions, dtype=np.float32)
        node.V = [np.empty(1, dtype=np.float32) for _ in self._actions]

        p, v = self._model(node.state)
        self.P = p.detach().numpy()

        self._child = []
        for action in self._actions:
            self._env.reset(options=self.state)

            state, _, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated

            node = self._nodeFactory.getOrCreate(state, info, done)

            self._child.append(node)

        return v.detach().value()

    def _backpropagation(self, node):
        pass
