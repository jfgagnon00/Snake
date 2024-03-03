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

        for _ in range(self._numIterations):
            node, trajectory = self._select(root)

            if node.done:
                r = 0
            else:
                v, mu, sigma = self._expand(node)
                r = mu + sigma * v

            # backpropagation(node)

        # newNode, intAction, newPolicy, value = None, 0, None, None

        # return newNode, \
        #     intAction, \
        #     newPolicy, \
        #     value

    def _select(self, node):
        trajectory = []

        while True:
            if node.done or node.isLeaf or node.vistCount >= self._maxVisitCount:
                break

            availableActions = node.state["available_actions"]
            availableActions = np.nonzero(availableActions)[0]

            node.vistCount += 1
            ucb = node.ucb(self._cpuct)
            ucb = ucb[availableActions]

            action = np.argmax(ucb)
            action = availableActions[action]
            trajectory.append((node, action))

            node = node.child[action]

        return node, trajectory

    def _expand(self, node):
        node.Q = np.zeros(self._numActions, dtype=np.float32)
        node.N = np.zeros(self._numActions, dtype=np.float32)
        node.V = [np.empty(1, dtype=np.float32) for _ in self._actions]

        availableActions = node.state["available_actions"]

        p, v = self._model(node.state)
        self.P = p.detach().numpy() * availableActions
        self.P = self.P / self.P.sum()

        self._child = []
        for action, available in zip(self._actions, availableActions):
            if available != 0:
                self._env.reset(options=self.state)

                state, _, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated

                node = self._nodeFactory.getOrCreate(state, info, done)

                self._child.append(node)
            else:
                self._child.append(None)

        return v.detach().value()

    def _backpropagation(self, node):
        pass
