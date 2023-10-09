from collections import deque


class GameSnake():
    """
    Represente le serpent
    """
    def __init__(self, position, direction):
        self.bodyParts = deque()

        # la tete est toujours le premier element
        self.bodyParts.append(position)

        self.bodyParts.append(position - direction)

        # la queue est toujours le dermier element
        self.bodyParts.append(position - direction - direction)

        self.direction = direction

    @property
    def head(self):
        return self.bodyParts[0]

    @property
    def tail(self):
        return self.bodyParts[-1]
