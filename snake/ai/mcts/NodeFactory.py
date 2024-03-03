from .Node import _Node


class _NodeFactory(object):
    def __init__(self):
        self._stateToNode = {}

    def getOrCreate(self, state, info, done):
        return _Node(state, info, done)
