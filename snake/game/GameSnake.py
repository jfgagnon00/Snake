import numpy as np

from collections import deque
from io import StringIO
from pprint import pprint
from snake.core import Vector


class GameSnake(object):
    """
    Represente le serpent
    """
    def __init__(self, headPosition=None, headDirection=None):
        self.bodyParts = deque()

        if headDirection and headPosition:
            # la tete est toujours le premier element
            self.bodyParts.append(headPosition)

            self.bodyParts.append(headPosition - headDirection)

            # la queue est toujours le dernier element
            self.bodyParts.append(headPosition - headDirection - headDirection)

    @property
    def head(self):
        return self.bodyParts[0]

    @property
    def tail(self):
        return self.bodyParts[-1]

    @property
    def length(self):
        return len(self.bodyParts)

    @property
    def direction(self):
        p0 = self.bodyParts[0]
        p1 = self.bodyParts[1]
        return p0 - p1

    def bodyPartsToNumpy(self):
        parts = [p.toNumpy() for p in self.bodyParts]
        return np.array(parts)

    def bodyPartsFromNumpy(self, bodyPartsNumpy):
        for bp in bodyPartsNumpy:
            p = Vector.fromNumpy(bp)
            self.bodyParts.append(p)

    def __str__(self):
        return "chiurre"
        return self.__repr__()

    def __repr__(self):
        with StringIO() as stream:
            pprint(vars(self), stream=stream)
            return stream.getvalue()

    def __eq__(self, other):
        if self.length != other.length:
            return False

        for a, b in zip(self.bodyParts, other.bodyParts):
            if a != b:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)
