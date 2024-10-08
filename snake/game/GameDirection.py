from enum import Enum
from snake.core import Vector

from .GameAction import GameAction


class GameDirection(Enum):
    """
    Les directions possibles pour la tete du serpent
    La convention est x == colonne, y == rangee
    """
    NORTH = Vector(0, -1)
    SOUTH = Vector(0,  1)
    EAST = Vector( 1, 0)
    WEST = Vector(-1, 0)
