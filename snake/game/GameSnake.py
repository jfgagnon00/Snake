import numpy as np

from collections import deque
from snake.core import Vector


class GameSnake(object):
    """
    Represente le serpent
    """
    def __init__(self, direction, position=None):
        self.bodyParts = deque()
        self.direction = direction

        if position:
            # la tete est toujours le premier element
            self.bodyParts.append(position)

            self.bodyParts.append(position - direction)

            # la queue est toujours le dernier element
            self.bodyParts.append(position - direction - direction)

    @property
    def head(self):
        return self.bodyParts[0]

    @property
    def tail(self):
        return self.bodyParts[-1]

    @property
    def length(self):
        return len(self.bodyParts)

    def bodyPartsToNumpy(self):
        parts = [p.toNumpy() for p in self.bodyParts]
        return np.array(parts)

    def bodyPartsFromNumpy(self, bodyPartsNumpy):
        for bp in bodyPartsNumpy:
            p = Vector.fromNumpy(bp)
            self.bodyParts.append(p)
