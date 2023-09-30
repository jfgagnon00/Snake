from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN
from pygame.key import get_pressed

from game.GameAction import GameAction


class InteractiveAgent():
    def __init__(self, lastKey=K_RIGHT):
        self._lastKey = lastKey
        self._keyMapping = {
            K_LEFT: GameAction.LEFT,
            K_RIGHT: GameAction.RIGHT,
            K_UP: GameAction.UP,
            K_DOWN: GameAction.DOWN
        }

    def getAction(self):
        keyPressed = get_pressed()

        for k, a in self._keyMapping.items():
            if keyPressed[k]:
                self._lastKey = k
                return a

        return self._keyMapping[self._lastKey]
