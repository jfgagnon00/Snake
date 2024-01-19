from enum import Enum

from snake.core import Vector


class GameAction(Enum):
    """
    Les actions possibles pour la tete du serpent
    """
    NORTH = Vector(0, -1)
    SOUTH = Vector(0,  1)
    EAST = Vector( 1, 0)
    WEST = Vector(-1, 0)
