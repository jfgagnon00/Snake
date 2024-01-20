from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN
from snake.ai.agents.AgentBase import AgentBase
from snake.core import Vector
from snake.game import GameAction


def _getAction(direction, action):
    if Vector.winding(direction, action.value) != 0:
        return action

    return GameAction(direction)

class AgentInteractive(AgentBase):
    """
    Specialization pour agent interactif.
    Note: specialiser pour play.py. Ne pas utiliser comme agent conventionel.
    """
    _KEY_HANDLERS = {
        K_LEFT: lambda d: _getAction(d, GameAction.WEST),
        K_RIGHT: lambda d: _getAction(d, GameAction.EAST),
        K_UP: lambda d: _getAction(d, GameAction.NORTH),
        K_DOWN: lambda d: _getAction(d, GameAction.SOUTH)
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

    def getAction(self, observations):
        """
        Obtenir action a partir de l'etat.
        """
        direction = observations["head_direction"]
        direction = Vector.fromNumpy(direction)

        if self._lastKeyDown in AgentInteractive._KEY_HANDLERS:
            handler = AgentInteractive._KEY_HANDLERS[self._lastKeyDown]
            return handler(direction)

        return GameAction(direction)
