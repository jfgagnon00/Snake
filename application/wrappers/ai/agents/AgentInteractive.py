from ai.agents.AgentBase import AgentBase
from game import GameAction
from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN


def _onLeft(direction):
    if direction.y > 0:
        # serpent va vers le bas, tourne a droite pour devenir la gauche
        return GameAction.TURN_CW

    if direction.y < 0:
        # serpent va vers le haut, tourne a gauche pour devenir la gauche
        return GameAction.TURN_CCW

    # serpent va vers la droite ou vers la gauche
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onRight(direction):
    if direction.y > 0:
        # serpent va vers le bas, tourne a gauche pour devenir la droite
        return GameAction.TURN_CCW

    if direction.y < 0:
        # serpent va vers le haut, tourne a droite pour devenir la droite
        return GameAction.TURN_CW

    # serpent va vers la droite ou vers la gauche
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onUp(direction):
    if direction.x > 0:
        # serpent va vers la droite, tourne a gauche pour devenir up
        return GameAction.TURN_CCW

    if direction.x < 0:
        # serpent va vers la gauche, tourne a droite pour devenir up
        return GameAction.TURN_CW

    # serpent va vers le haut ou vers la bas
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onDown(direction):
    if direction.x > 0:
        # serpent va vers la droite, tourne a droite pour devenir up
        return GameAction.TURN_CW

    if direction.x < 0:
        # serpent va vers la gauche, tourne a gauche pour devenir up
        return GameAction.TURN_CCW

    # serpent va vers le haut ou vers la bas
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

class AgentInteractive(AgentBase):
    """
    Specialization pour agent interactif.
    Note: specialiser pour play.py. Ne pas utiliser comme agent conventionel.
    """
    _KEY_HANDLERS = {
        K_LEFT: _onLeft,
        K_RIGHT: _onRight,
        K_UP: _onUp,
        K_DOWN: _onDown
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
        if key in AgentInteractive._KEY_HANDLERS:
            self._lastKeyDown = key

    def getAction(self, direction):
        """
        Obtenir action a partir de l'etat.
        """
        if self._lastKeyDown in AgentInteractive._KEY_HANDLERS:
            handler = AgentInteractive._KEY_HANDLERS[self._lastKeyDown]
            return handler(direction)

        return GameAction.FORWARD
