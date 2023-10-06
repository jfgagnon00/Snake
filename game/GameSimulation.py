import numpy as np
import random

from core import Vector
from .GameAction import GameAction
from .GameSnake import GameSnake
from .GridOccupancy import GridOccupancy


class GameSimulation():
    def __init__(self, simulationConfig):
        self._gridWidth = simulationConfig.gridWidth
        self._gridHeight = simulationConfig.gridHeight
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
        self._grid = np.zeros(shape=shape, dtype=np.uint8)
        self._snake = GameSnake(Vector(4, 1), Vector(1, 0))

        # placer le serpent dans la grille
        self._setSnakeInGrid(True)

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
            self._setSnakeInGrid(False)
            self._snake.bodyParts.appendleft(head)
            self._setSnakeInGrid(True)
            self._placeFood()
            self._score += 1
            return False

        if head.x < 0 or \
           head.y < 0 or \
           head.x >= self._gridWidth or \
           head.y >= self._gridHeight or \
           self._grid[head.x, head.y] != GridOccupancy.EMPTY:
            # tete est en collision ou en dehors de la grille, terminer
            return True

        # bouger le corps du serpent
        self._setSnakeInGrid(False)
        self._snake.bodyParts.pop()
        self._snake.bodyParts.appendleft(head)
        self._setSnakeInGrid(True)

        return False

    def _setSnakeInGrid(self, show):
        # sous optimal, a changer
        if show:
            for i, p in enumerate(self._snake.bodyParts):
                value = GridOccupancy.SNAKE_HEAD if i == 0 else GridOccupancy.SNAKE_BODY
                self._grid[p.x, p.y] = value
        else:
            for i in self._snake.bodyParts:
                self._grid[i.x, i.y] = GridOccupancy.EMPTY

    def _placeFood(self):
        # sous optimal, a changer
        while True:
            x = random.randint(0, self._gridWidth - 1)
            y = random.randint(0, self._gridHeight - 1)

            if self._grid[x, y] == 0:
                self._food = Vector(x, y)
                self._grid[x, y] = GridOccupancy.FOOD
                break
