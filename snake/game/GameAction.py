from strenum import StrEnum


class GameAction(StrEnum):
    """
    Les actions possibles pour la tete du serpent
    """
    TURN_CW = "TURN_CW"
    TURN_CCW = "TURN_CCW"
    FORWARD = "FORWARD"

    @property
    def krot90(self):
        if self == GameAction.TURN_CW:
            return -1

        if self == GameAction.TURN_CCW:
            return 1

        return 0

    @property
    def flip(self):
        if self == GameAction.TURN_CCW:
            return GameAction.TURN_CW

        if self == GameAction.TURN_CW:
            return GameAction.TURN_CCW

        return GameAction.FORWARD
