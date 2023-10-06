from enum import IntEnum

class GameAction(IntEnum):
    """
    Les actions possibles pour la tete du serpent
    """
    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2
    COUNT = 3
