from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN

from game.GameAction import GameAction


def _onLeft(direction):
    if direction.y > 0:
        # serpent va vers le bas, tourne a droite pour devenir la gauche
        return GameAction.TURN_RIGHT

    if direction.y < 0:
        # serpent va vers le haut, tourne a gauche pour devenir la gauche
        return GameAction.TURN_LEFT

    # serpent va vers la droite ou vers la gauche
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onRight(direction):
    if direction.y > 0:
        # serpent va vers le bas, tourne a gauche pour devenir la droite
        return GameAction.TURN_LEFT

    if direction.y < 0:
        # serpent va vers le haut, tourne a droite pour devenir la droite
        return GameAction.TURN_RIGHT

    # serpent va vers la droite ou vers la gauche
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onUp(direction):
    if direction.x > 0:
        # serpent va vers la droite, tourne a gauche pour devenir up
        return GameAction.TURN_LEFT

    if direction.x < 0:
        # serpent va vers la gauche, tourne a droite pour devenir up
        return GameAction.TURN_RIGHT

    # serpent va vers le haut ou vers la bas
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onDown(direction):
    if direction.x > 0:
        # serpent va vers la droite, tourne a droite pour devenir up
        return GameAction.TURN_RIGHT

    if direction.x < 0:
        # serpent va vers la gauche, tourne a gauche pour devenir up
        return GameAction.TURN_LEFT

    # serpent va vers le haut ou vers la bas
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

class InteractiveAgent():
    _KEY_HANDLERS = {
        K_LEFT: _onLeft,
        K_RIGHT: _onRight,
        K_UP: _onUp,
        K_DOWN: _onDown
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self._lastKeyDown = -1

    def onKeyDown(self, key):
        if key in InteractiveAgent._KEY_HANDLERS:
            self._lastKeyDown = key

    def getAction(self, direction):
        if self._lastKeyDown in InteractiveAgent._KEY_HANDLERS:
            handler = InteractiveAgent._KEY_HANDLERS[self._lastKeyDown]
            return handler(direction)

        return GameAction.FORWARD
