import numpy as np
import random

from core.Vector import Vector
from .GameAction import GameAction
from .Snake import Snake


class GameEnvironment():
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

    @property
    def score(self):
        return self._score

    def reset(self):
        self._score = 0

        shape = (self._gridWidth, self._gridHeight)
        self._grid = np.zeros(shape=shape, dtype=np.int8)
        self._snake = Snake(Vector(4, 1), Vector(1, 0))

        # placer le serpent dans la grille
        self._setSnakeInGrid(1)

        self._placeFood()

    def apply(self, action):
        # d est la meme instance que le serpent
        # la mise a jour va modifier le serpent aussi
        d = self._snake.direction

        if action == GameAction.TURN_LEFT:
            # tourne direction 90 degres CCW
            d.x, d.y = d.y, -d.x

        if action == GameAction.TURN_RIGHT:
            # tourne direction 90 degres CW
            d.x, d.y = -d.y, d.x

        # bouger la tete dans la nouvelle direction
        # ATTENTION: l'operateur + cree une nouvelle instance
        head = self._snake.head + d

        if False:
            print("HEAD:", head.x, head.y, "DIR", d.x, d.y)

        if head == self._food:
            # tete est sur la nourriture, grandire le serpent
            self._setSnakeInGrid(0)
            self._snake.bodyParts.appendleft(head)
            self._setSnakeInGrid(1)
            self._placeFood()
            self._score += 1
            return False

        if head.x < 0 or \
           head.y < 0 or \
           head.x >= self._gridWidth or \
           head.y >= self._gridHeight or \
           self._grid[head.x, head.y] == 1:
            # tete est en collision ou en dehors de la grille, terminer
            return True

        # bouger le corps du serpent
        self._setSnakeInGrid(0)
        self._snake.bodyParts.pop()
        self._snake.bodyParts.appendleft(head)
        self._setSnakeInGrid(1)

        return False

    def _setSnakeInGrid(self, value):
        # sous optimal, a changer
        for i in self._snake.bodyParts:
            self._grid[i.x, i.y] = value

    def _placeFood(self):
        # sous optimal, a changer
        while True:
            x = random.randint(0, self._gridWidth - 1)
            y = random.randint(0, self._gridHeight - 1)

            if self._grid[x, y] == 0:
                self._food = Vector(x, y)
                self._grid[x, y] = 1
                break
