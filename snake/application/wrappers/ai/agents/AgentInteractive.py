from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN
from snake.ai.agents.AgentBase import AgentBase
from snake.core import Vector
from snake.game import GameAction, GameDirection


def _getAction(currentDirection, desirecDirection):
    w = Vector.winding(currentDirection, desirecDirection)
    if w == 0:
        return GameAction.FORWARD

    if w == 1:
        return GameAction.TURN_CCW

    return GameAction.TURN_CW

class AgentInteractive(AgentBase):
    """
    Specialization pour agent interactif.
    Note: specialiser pour play.py. Ne pas utiliser comme agent conventionel.
    """
    _KEY_HANDLERS = {
        K_LEFT: lambda d: _getAction(d, GameDirection.WEST.value),
        K_RIGHT: lambda d: _getAction(d, GameDirection.EAST.value),
        K_UP: lambda d: _getAction(d, GameDirection.NORTH.value),
        K_DOWN: lambda d: _getAction(d, GameDirection.SOUTH.value)
    }

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self._lastKeyDown = -1

    def onKeyDown(self, key):
        """
        Handler système d'input
        """
        self._lastKeyDown = key

    def getAction(self, observations, infos):
        """
        Obtenir action a partir de l'etat.
        """
        direction = infos["simulation_state"].snake.direction

        if self._lastKeyDown in AgentInteractive._KEY_HANDLERS:
            handler = AgentInteractive._KEY_HANDLERS[self._lastKeyDown]
            return handler(direction)

        return GameAction.FORWARD
