from enum import Enum

class GameAction(Enum):
    """
    Les actions possibles pour la tete du serpent
    """
    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2
