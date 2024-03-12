import hashlib

from io import BytesIO
from pickle import dump

from .Node import _Node


class _NodeFactory(object):
    def __init__(self):
        self._stateToNode = {}

    def clear(self):
        self._stateToNode.clear()

    def getOrCreate(self, state, info, done, won):
        h = _NodeFactory._hash(state, info)
        node = self._stateToNode.get(h, None)
        if node is None:
            node = _Node(state, info, done, won)
            self._stateToNode[h] = node
        else:
            pass

        return node

    def validateVisitCount(self):
        for n in self._stateToNode.values():
            assert n.vistCount == 0

    @staticmethod
    def _hash(state, info):
        with BytesIO() as stream:
            # state est completement defini par les positions du serpent et la direction de la tete
            dump(state["occupancy_grid"], file=stream)
            dump(state["head_direction"], file=stream)

            return hashlib.md5(stream.getbuffer()).hexdigest()
