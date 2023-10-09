import numpy as np
import random

from core import Delegate, Vector
from .GameAction import GameAction
from .GameSnake import GameSnake
from .GridOccupancy import GridOccupancy


class GameSimulation():
    """
    Responsable de la logique de simulation du jeu de serpent.
    La simulation evolue sur une grille discrete.
    """
    def __init__(self, simulationConfig):
        self._occupancyGridWidth = max(5, simulationConfig.gridWidth)
        self._occupancyGridHeight = max(5, simulationConfig.gridHeight)

        self._outOfBoundsDelegate = Delegate()
        self._collisionDelegate = Delegate()
        self._eatDelegate = Delegate()
        self._winDelegate = Delegate()
        self._turnDelegate = Delegate()
        self._moveDelegate = Delegate()

        self.reset()

    @property
    def snake(self):
        return self._snake

    @property
    def food(self):
        return self._food

    @property
    def occupancyGrid(self):
        return self._occupancyGrid

    @property
    def outOfBoundsDelegate(self):
        return self._outOfBoundsDelegate

    @property
    def collisionDelegate(self):
        return self._collisionDelegate

    @property
    def eatDelegate(self):
        return self._eatDelegate

    @property
    def winDelegate(self):
        return self._winDelegate

    @property
    def turnDelegate(self):
        return self._turnDelegate

    @property
    def moveDelegate(self):
        return self._moveDelegate

    @property
    def score(self):
        return self._score

    def reset(self):
        """
        Reinitialize etats internes
        """
        self._score = 0

        shape = (self._occupancyGridWidth, self._occupancyGridHeight)
        self._occupancyGrid = np.zeros(shape=shape, dtype=np.uint8)
        self._snake = GameSnake(Vector(4, 1), Vector(1, 0))

        # placer le serpent dans la grille
        self._setSnakeInGrid(True)

        self._placeFood()

    def apply(self, action):
        """
        Met a jour la simulation en fonction de l'action fournie.
        """
        # d est la meme instance que le serpent
        # la mise a jour va modifier le serpent aussi
        d = self._snake.direction

        if action == GameAction.TURN_LEFT:
            # tourne direction 90 degres CCW
            d.x, d.y = d.y, -d.x
            self._turnDelegate()

        if action == GameAction.TURN_RIGHT:
            # tourne direction 90 degres CW
            d.x, d.y = -d.y, d.x
            self._turnDelegate()

        # bouger la tete dans la nouvelle direction
        # ATTENTION: l'operateur + cree une nouvelle instance
        head = self._snake.head + d

        if head == self._food:
            # tete est sur la nourriture, grandire le serpent
            self._setSnakeInGrid(False)
            self._snake.bodyParts.appendleft(head)
            self._setSnakeInGrid(True)
            self._score += 1

            cellCount = self._occupancyGridWidth * self._occupancyGridHeight
            if len(self._snake.bodyParts) == cellCount:
                self._food = None
                # serpent couvre toutes les cellules, gagner la partie
                self._winDelegate()
                return True
            else:
                self._placeFood()
                self._eatDelegate()
                return False

        if head.x < 0 or \
           head.y < 0 or \
           head.x >= self._occupancyGridWidth or \
           head.y >= self._occupancyGridHeight:
           # tete est en dehors de la grille, terminer
           self._outOfBoundsDelegate()
           return True

        # la tete va bouger, donc la queue aussi
        # pas de collision possible avec la queue
        if head != self._snake.tail and \
           self._occupancyGrid[head.x, head.y] != GridOccupancy.EMPTY:
            # tete est en collision
            self._collisionDelegate()
            return True

        # bouger le corps du serpent
        self._setSnakeInGrid(False)
        self._snake.bodyParts.pop()
        self._snake.bodyParts.appendleft(head)
        self._setSnakeInGrid(True)
        self._moveDelegate()

        return False

    def _setSnakeInGrid(self, show):
        # sous optimal, a changer
        value = GridOccupancy.SNAKE_BODY if show else GridOccupancy.EMPTY

        for i, p in enumerate(self._snake.bodyParts):
            self._occupancyGrid[p.x, p.y] = value

        if show:
            head = self._snake.head
            self._occupancyGrid[head.x, head.y] = GridOccupancy.SNAKE_HEAD

            tail = self._snake.tail
            self._occupancyGrid[tail.x, tail.y] = GridOccupancy.SNAKE_TAIL

    def _placeFood(self):
        # sous optimal, a changer
        while True:
            x = random.randint(0, self._occupancyGridWidth - 1)
            y = random.randint(0, self._occupancyGridHeight - 1)

            if self._occupancyGrid[x, y] == 0:
                self._food = Vector(x, y)
                self._occupancyGrid[x, y] = GridOccupancy.FOOD
                break
