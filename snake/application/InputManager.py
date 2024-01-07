from core import Delegate
from pygame import QUIT, KEYDOWN, KEYUP
from pygame.event import get as py_eventGet

class _InputManager():
    def __init__(self):
        self._keyDownDelegate = Delegate()
        self._quitDelegate = Delegate()
        self._anyKeyPressedDelegate = Delegate()

    @property
    def keyDownDelegate(self):
        return self._keyDownDelegate

    @property
    def quitDelegate(self):
        return self._quitDelegate

    @property
    def anyKeyPressedDelegate(self):
        return self._anyKeyPressedDelegate

    def update(self):
        for e in py_eventGet():
            if e.type == QUIT:
                self._quitDelegate()

            if e.type == KEYDOWN:
                self._keyDownDelegate(e.key)

            if e.type == KEYUP:
                self._anyKeyPressedDelegate()
