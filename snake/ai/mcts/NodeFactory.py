import hashlib

from io import BytesIO
from pickle import dump

from .Node import _Node


class _NodeFactory(object):
    def __init__(self):
        self._stateToNode = {}

    def getOrCreate(self, state, info, done):
        h = _NodeFactory._hash(state, info, done)
        if h in self._stateToNode:
            return self._stateToNode[h]

        node = _Node(state, info, done)
        self._stateToNode[h] = node

        return node

    @staticmethod
    def _hash(state, info, done):
        with BytesIO() as stream:
            for k in _Node.stateKeys():
                dump(state[k], file=stream)

            for k in _Node.infoKeys():
                dump(info[k], file=stream)

            dump(done, file=stream)

            return hashlib.md5(stream.getbuffer()).hexdigest()
