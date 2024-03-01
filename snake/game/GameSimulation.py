import numpy as np

from io import StringIO
from pprint import pprint
from snake.core import Delegate, Vector, RandomProxy
from .GameAction import GameAction
from .GameDirection import GameDirection
from .GameSnake import GameSnake
from .GridOccupancy import GridOccupancy


class ResetException(Exception):
    def __init__(self, message, occupancyGrid, position):
        super().__init__(message)
        self.occupancyGrid = occupancyGrid
        self.position = position

    def __str__(self):
        with StringIO() as stream:
            print(file=stream)
            print(super().__str__(), file=stream)
            if self.occupancyGrid is None:
                print("No occupancy grid", file=stream)
            else:
                print("occupancyGrid shape", self.occupancyGrid.shape, file=stream)
                print("occupancyGrid", file=stream)
                pprint(self.occupancyGrid, stream=stream)
            print("Position", self.position, file=stream)
            return stream.getvalue()

class GameSimulation(object):
    """
    Responsable de la logique de simulation du jeu de serpent.
    La simulation evolue sur une grille discrete.
    """
    def __init__(self, simulationConfig):
        self._occupancyGridWidth = max(6, simulationConfig.gridWidth)
        self._occupancyGridHeight = max(6, simulationConfig.gridHeight)

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
    def occupancyGridCount(self):
        """
        Retourne une reference sur le nombre de foix une cellule a ete visitee.
        Attention, prendre la convention d'acces numpy
        """
        return self._occupancyGridCount

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

    def reset(self, options=None):
        """
        Reinitialize etats internes
        """
        # numpy est transpose par rapport au sens naturel
        # l'acces a _occupancyGrid suivra la convention numpy
        shape = (self._occupancyGridHeight, self._occupancyGridWidth)
        self._occupancyGridCount = np.zeros(shape=shape, dtype=np.int32)
        self._occupancyGrid = np.zeros(shape=shape, dtype=np.int32)

        try:
            food = options["food_position"]
            direction = options["head_direction"]
            bodyparts = options["snake_bodyparts"]

            self._score = options["score"]
            self._snake = GameSnake(Vector.fromNumpy(direction))
            self._snake.bodyPartsFromNumpy(bodyparts)
        except:
            food = None
            self._score = 0
            self._snake = GameSnake(GameDirection.EAST.value, position=Vector(3, 1))

        # placer le serpent dans la grille
        self._setSnakeInGrid(True)

        if food is None:
            self._placeRandomFood()
        else:
            self._food = Vector.fromNumpy(food)
            self._setFoodInGrid(True)

    def apply(self, action):
        """
        Met a jour la simulation en fonction de l'action fournie.
        """
        if action != GameAction.FORWARD:
            k = action.krot90
            self._snake.direction = self._snake.direction.rot90(k)
            self._turnDelegate()

        # bouger la tete dans la nouvelle direction
        # ATTENTION: l'operateur + cree une nouvelle instance
        head = self._snake.head + self._snake.direction

        if head == self._food:
            self._setFoodInGrid(False)

            # tete est sur la nourriture, grandire le serpent
            self._setSnakeInGrid(False)
            self._snake.bodyParts.appendleft(head)
            self._setSnakeInGrid(True)
            self._score += 1

            self._occupancyGridCount = np.zeros(shape=(self._occupancyGridHeight, self._occupancyGridWidth), dtype=np.int32)

            cellCount = self._occupancyGridWidth * self._occupancyGridHeight
            if len(self._snake.bodyParts) == cellCount:
                self._food = None
                # serpent couvre toutes les cellules, gagner la partie
                self._winDelegate()
            else:
                self._placeRandomFood()
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

    def getObservations(self):
        """
        Retourne les etats internes sous forme standardisee
        """
        return {
            # shape est (Channel, Height, Width)
            "occupancy_grid": np.expand_dims(self.occupancyGrid, axis=0).copy(),
            "occupancy_heatmap": np.expand_dims(self._occupancyGridCount, axis=0).copy(),
            "head_direction": self.snake.direction.toNumpy(),
            "head_position": self.snake.head.toNumpy(),
            "food_position": None if self.food is None else self.food.toNumpy(),
            "length": self.snake.length,
            "score": self.score,
        }

    def getInfo(self):
        """
        Retourne informations supplementaires
        """
        return {
            "snake_bodyparts": self.snake.bodyPartsToNumpy(),
        }

    def _setFoodInGrid(self, show):
        # validation
        if not (0 <= self._food.x and self._food.x < self._occupancyGridWidth):
            raise ResetException("Positon x food invalide", None, self._food.copy())

        if not (0 <= self._food.y and self._food.y < self._occupancyGridHeight):
            raise ResetException("Positon y food invalide", None, self._food.copy())

        expectedOccupancy = GridOccupancy.EMPTY if show else GridOccupancy.FOOD
        if self._occupancyGrid[self._food.y, self._food.x] != expectedOccupancy:
            raise ResetException("Occupancy food invalide",
                                 self._occupancyGrid.copy(),
                                 self._food.copy())

        # placer dans la grille
        self._occupancyGrid[self._food.y, self._food.x] = GridOccupancy.FOOD if show else GridOccupancy.EMPTY

    def _setSnakeInGrid(self, show):
        # sous optimal, a changer
        value = GridOccupancy.SNAKE_BODY if show else GridOccupancy.EMPTY

        for i, p in enumerate(self._snake.bodyParts):
            if show and self._occupancyGrid[p.y, p.x] != GridOccupancy.EMPTY:
                raise ResetException("Placement snake bodyParts invalide",
                                     self._occupancyGrid.copy(),
                                     p.copy())

            self._occupancyGrid[p.y, p.x] = value

        if show:
            tail = self._snake.tail

            if self._occupancyGrid[tail.y, tail.x] != GridOccupancy.SNAKE_BODY:
                raise ResetException("Placement snake tail invalide",
                                     self._occupancyGrid.copy(),
                                     tail.copy())

            self._occupancyGrid[tail.y, tail.x] = GridOccupancy.SNAKE_TAIL

            head = self._snake.head

            if self._occupancyGrid[head.y, head.x] != GridOccupancy.SNAKE_BODY:
                raise ResetException("Placement snake head invalide",
                                     self._occupancyGrid.copy(),
                                     head.copy())

            self._occupancyGrid[head.y, head.x] = GridOccupancy.SNAKE_HEAD
            self._occupancyGridCount[head.y, head.x] += 1

    def _placeRandomFood(self):
        # trouver les cellules libres a partir de _occupancyGrid
        allCells = np.arange(self._occupancyGridWidth * self._occupancyGridHeight)
        freeCells = np.where(self._occupancyGrid == GridOccupancy.EMPTY, True, False)

        if len(freeCells) == 0:
            self._food = None
            return

        freeCells = allCells[freeCells.reshape(-1)]

        # prendre une cellule au hasard
        cellIndex = RandomProxy.choice(freeCells)

        # transformer index en coordonnees
        x = cellIndex % self._occupancyGridWidth
        y = cellIndex // self._occupancyGridWidth

        self._food = Vector(x, y)
        self._setFoodInGrid(True)
