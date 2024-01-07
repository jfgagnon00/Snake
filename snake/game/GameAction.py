from enum import StrEnum, auto

class GameAction(StrEnum):
    """
    Les actions possibles pour la tete du serpent
    """
    TURN_CCW = auto()
    TURN_CW = auto()
    FORWARD = auto()
