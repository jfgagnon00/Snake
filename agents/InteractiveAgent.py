from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN

from game.GameAction import GameAction


def _onLeft(direction):
    if direction.y > 0:
        # serpent va vers le haut, tourne a gauche pour devenir la gauche
        return GameAction.TURN_LEFT

    if direction.y < 0:
        # serpent va vers le bas, tourne a droite pour devenir la gauche
        return GameAction.TURN_RIGHT

    # serpent va vers la droite ou vers la gauche
    # ne pas permettre un retour en arriere
    # donc garder la direction
    return GameAction.FORWARD

def _onRight(direction):
    if direction.y > 0:
        # serpent va vers le haut, tourne a droite pour devenir la droite
        return GameAction.TURN_RIGHT

    if direction.y < 0:
        # serpent va vers le bas, tourne a gauche pour devenir la droite
        return GameAction.TURN_LEFT

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
        self._isKeyDown = {
            K_LEFT: False,
            K_RIGHT: False,
            K_UP: False,
            K_DOWN: False
        }

    def onKeyDown(self, key):
        if key in self._isKeyDown:
            self._isKeyDown[key] = True

    def onKeyUp(self, key):
        if key in self._isKeyDown:
            self._isKeyDown[key] = False

    def getAction(self, direction):
        for k, handler in InteractiveAgent._KEY_HANDLERS.items():
            if self._isKeyDown[k]:
                return handler(direction)

        return GameAction.FORWARD
