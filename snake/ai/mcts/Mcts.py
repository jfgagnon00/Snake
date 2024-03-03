import numpy as np

from copy import deepcopy
from .Node import _Node
from .NodeException import _NodeException
from .NodeFactory import _NodeFactory

class _Mcts(object):
    def __init__(self, trainConfig):
        self._cpuct = trainConfig.mcts.cpuct
        self._numIterations = trainConfig.mcts.numIterations
        self._maxVisitCount = trainConfig.maxVisitCount
        self._nodeFactory = _NodeFactory()

    def initEnv(self, env):
        self._env = deepcopy(env)

    def reset(self):
        pass

    def search(root, env, state, info, model):
        # if root is None:
        #     raise _NodeException("None root")

        # root.valiate(state, info)

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

    def _backpropagation(self, node):
        pass
