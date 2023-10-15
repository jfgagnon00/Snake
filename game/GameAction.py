from enum import StrEnum, auto

class GameAction(StrEnum):
    """
    Les actions possibles pour la tete du serpent
    """
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    FORWARD = auto()

    @classmethod
    def fromInt(cls, intValue):
        if not hasattr(cls, "_ACTIONS_VALUES"):
            cls._ACTIONS_VALUES = list(GameAction)

        return GameAction(cls._ACTIONS_VALUES[intValue])
