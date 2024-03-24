import hashlib

from io import BytesIO
from pickle import dump
from snake.core import Profile

from .Node import _Node


class _NodeFactory(object):
    def __init__(self):
        self._stateToNode = {}
        self.getOrCreateDuration = 0

    def clear(self):
        self._stateToNode.clear()

    def getOrCreate(self, state, info, done, won):
        with Profile() as p:
            h = _NodeFactory._hash(state, info)
            node = self._stateToNode.get(h, None)
            if node is None:
                node = _Node(state, info, done, won)
                self._stateToNode[h] = node
            else:
                # validation collision dans le hash
                node.validate(state, info)

        self.getOrCreateDuration = p.duration

        return node

    def validateVisitCount(self):
        for n in self._stateToNode.values():
            assert n.vistCount == 0

    @staticmethod
    def _hash(state, info):
        with BytesIO() as stream:
            dump(info["simulation_state"].snake.bodyParts, file=stream)
            dump(info["simulation_state"].food, file=stream)

            return hashlib.md5(stream.getbuffer()).hexdigest()
