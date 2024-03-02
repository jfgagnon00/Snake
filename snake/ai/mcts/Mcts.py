import numpy as np

from .Node import _Node
from .NodeException import _NodeException

class _Mcts(object):
    def __init__(self, trainConfig):
        self._cpuct = trainConfig.mcts.cpuct
        self._numIterations = trainConfig.mcts._numIterations
        self._maxVisitCount = trainConfig.maxVisitCount

    def _select(self, node):
        while not node.isLeaf:
            ucb = node.ucb()
            index = np.argmax(ucb)
            node = node.child[index]

        return node

    def _backpropagation(self, node):
        pass

    def getAction(root, env, state, info, model):
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
