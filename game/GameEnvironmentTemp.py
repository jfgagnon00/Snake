import numpy as np
import random

from .Point import Point
from .Snake import Snake


class GameEnvironmentTemp():
    def __init__(self, gameConfig):
        self._gridWidth = gameConfig.grid_width
        self._gridHeight = gameConfig.grid_height
        self.reset()

    @property
    def snake(self):
        return self._snake

    @property
    def food(self):
        return self._food

    def reset(self):
        shape = (self._gridHeight, self._gridWidth)
        self._grid = np.zeros(shape=shape, dtype=np.int8)
        self._snake = Snake(4, 1)

        # placer le serpent dans la grille
        self._setSnakeInGrid(1)

        self._placeFood()

    def apply(self, action):
        self._setSnakeInGrid(0)
        self._snake.move(action)
        # self._setSnakeInGrid(1)

        head = self._snake.head
        if head.x < 0 or \
           head.y < 0 or \
           head.x >= self._gridWidth or \
           head.y >= self._gridHeight:
            return True

        return False

    def _setSnakeInGrid(self, value):
        # sous optimal, a changer
        for i in self._snake.bodyParts:
            self._grid[i.y, i.x] = value

    def _placeFood(self):
        while True:
            x = random.randint(0, self._gridWidth - 1)
            y = random.randint(0, self._gridHeight - 1)

            if self._grid[y, x] == 0:
                self._food = Point(x, y)
                self._grid[y, x] = 1
                break
