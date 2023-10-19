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
        """
        Retourne une reference sur la grille d'occupation.
        Attention, prendre la convention d'acces numpy
        """
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

        # numpy est transpose par rapport au sens naturel
        # l'acces a _occupancyGrid suivra la convention numpy
        shape = (self._occupancyGridHeight, self._occupancyGridWidth)
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

        if action == GameAction.TURN_CCW:
            # tourne direction 90 degres CCW
            d.x, d.y = d.y, -d.x
            self._turnDelegate()

        if action == GameAction.TURN_CW:
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
            else:
                self._placeFood()
                self._eatDelegate()

            return

        if head.x < 0 or \
           head.y < 0 or \
           head.x >= self._occupancyGridWidth or \
           head.y >= self._occupancyGridHeight:
           # tete est en dehors de la grille, terminer
           self._outOfBoundsDelegate()
           return

        # la tete va bouger, donc la queue aussi
        # pas de collision possible avec la queue
        if head != self._snake.tail and \
           self._occupancyGrid[head.y, head.x] != GridOccupancy.EMPTY:
            # tete est en collision
            self._collisionDelegate()
            return

        # bouger le corps du serpent
        self._setSnakeInGrid(False)
        self._snake.bodyParts.pop()
        self._snake.bodyParts.appendleft(head)
        self._setSnakeInGrid(True)
        self._moveDelegate()

    def _setSnakeInGrid(self, show):
        # sous optimal, a changer
        value = GridOccupancy.SNAKE_BODY if show else GridOccupancy.EMPTY

        for i, p in enumerate(self._snake.bodyParts):
            self._occupancyGrid[p.y, p.x] = value

        if show:
            head = self._snake.head
            self._occupancyGrid[head.y, head.x] = GridOccupancy.SNAKE_HEAD

            tail = self._snake.tail
            self._occupancyGrid[tail.y, tail.x] = GridOccupancy.SNAKE_TAIL

    def _placeFood(self):
        # trouver les cellules libres a partir de _occupancyGrid
        allCells = np.arange(self._occupancyGridWidth * self._occupancyGridHeight)
        freeCells = np.where(self._occupancyGrid == GridOccupancy.EMPTY, True, False)
        freeCells = allCells[freeCells.reshape(-1)]

        # prendre une cellule au hasard
        cellIndex = random.choice(freeCells)

        # transformer index en coordonnees
        x = cellIndex % self._occupancyGridWidth
        y = cellIndex // self._occupancyGridWidth

        # validation
        assert 0 <= x and x < self._occupancyGridWidth
        assert 0 <= y and y < self._occupancyGridHeight
        assert self._occupancyGrid[y, x] == GridOccupancy.EMPTY

        # placer dans la grille
        self._food = Vector(x, y)
        self._occupancyGrid[y, x] = GridOccupancy.FOOD
