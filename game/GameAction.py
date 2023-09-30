from enum import Enum

class GameAction(Enum):
    """
    Les actions possibles pour la tete du serpent
    """
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3