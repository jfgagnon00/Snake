import numpy as np

from copy import deepcopy
from io import StringIO
from pprint import pprint
from snake.core import Vector, RandomProxy
from .GameSnake import GameSnake
from .GameDirection import GameDirection


class GameSimulationState(object):
    def __init__(self, simulationConfig):
        self._gridWidth = simulationConfig.gridWidth
        self._gridHeight = simulationConfig.gridHeight
        self._snake = None
        self._food = None

    @property
    def gridWidth(self):
        return self._gridWidth

    @property
    def gridHeight(self):
        return self._gridHeight

    @property
    def gridShape(self):
        # numpy est transpose par rapport au sens naturel
        # l'acces a _occupancyGrid suivra la convention numpy
        return (self._gridHeight, self._gridWidth)

    @property
    def snake(self):
        return self._snake

    @property
    def food(self):
        return self._food

    @food.setter
    def food(self, value):
        self._food = value

    @property
    def score(self):
        return self._snake.length - 3

    def copy(self):
        return deepcopy(self)

    def outOfBound(self, position):
        return position.x < 0 or \
                position.y < 0 or \
                position.x >= self.gridWidth or \
                position.y >= self.gridHeight

    def collisonTest(self, position):
        # pas de collision possible avec la queue
        s = self._snake
        for i in range(s.length - 1):
            if position == self.snake.bodyParts[i]:
                return True

        return False

    def availableActions(self):
        head_p = self._snake.head

        f = self._snake.direction
        cw = Vector.rot90(f, -1)
        ccw = Vector.rot90(f, 1)
        head_cw = head_ccw = head_f = 0

        p = head_p + f
        if not (self.outOfBound(p) or self.collisonTest(p)):
            head_f = 1

        p = head_p + cw
        if not (self.outOfBound(p) or self.collisonTest(p)):
            head_cw = 1

        p = head_p + ccw
        if not (self.outOfBound(p) or self.collisonTest(p)):
            head_ccw = 1

        # doit suivre l'ordre de GameAction
        return (head_cw, head_ccw, head_f)

    @staticmethod
    def initRandom(simulationState):
        simulationState._snake = GameSnake(headPosition=Vector(3, 1),
                                           headDirection=GameDirection.EAST.value)
        simulationState._food = GameSimulationState.randomFood(simulationState)

    @staticmethod
    def randomFood(simulationState):
        freeCells = np.full(simulationState.gridShape, True)
        for p in simulationState.snake.bodyParts:
            freeCells[p.y, p.x] = False

        if len(freeCells) == 0:
            return None

        allCells = np.arange(simulationState.gridWidth * simulationState.gridHeight)
        freeCells = allCells[freeCells.reshape(-1)]

        # prendre une cellule au hasard
        cellIndex = RandomProxy.choice(freeCells)

        # transformer index en coordonnees
        x = cellIndex % simulationState.gridWidth
        y = cellIndex // simulationState.gridHeight

        return Vector(x, y)

    def __repr__(self):
        with StringIO() as stream:
            pprint(vars(self), stream=stream)
            return stream.getvalue()

    def __eq__(self, other):
        if self._gridWidth != other._gridWidth:
            return False

        if self._gridHeight != other._gridHeight:
            return False

        if self._food != other._food:
            return False

        return self._snake == other._snake

    def __ne__(self, other):
        return not self.__eq__(other)
