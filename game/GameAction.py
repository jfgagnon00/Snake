from enum import StrEnum, auto

class GameAction(StrEnum):
    """
    Les actions possibles pour la tete du serpent
    """
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    FORWARD = auto()
    COUNT = auto()
