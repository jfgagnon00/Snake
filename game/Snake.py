from collections import deque
from .GameAction import GameAction
from .Point import Point

class Snake():
    """
    Represente le serpent
    """
    def __init__(self, x, y):
        self.bodyParts = deque()

        # la tete est toujours le premier element
        self.bodyParts.append(Point(x, y))

        self.bodyParts.append(Point(x - 1, y))

        # la queue est toujours le dermier element
        self.bodyParts.append(Point(x - 2, y))

    @property
    def head(self):
        return self.bodyParts[0]

    def move(self, action):
        h = self.head
        self.bodyParts.pop()
        if action == GameAction.RIGHT:
            h = Point(h.x + 1, h.y)
        elif action == GameAction.LEFT:
            h = Point(h.x - 1, h.y)
        elif action == GameAction.DOWN:
            h = Point(h.x, h.y + 1)
        elif action == GameAction.UP:
            h = Point(h.x, h.y - 1)

        self.bodyParts.appendleft(h)
