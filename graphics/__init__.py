"""
Gere l'afficahge de la simulation
"""

from pygame import init as py_init, quit as py_quit
from pygame.event import pump as py_eventPump
from pygame.font import init as py_fontInit, quit as py_fontQuit

from .GraphicWindow import GraphicWindow


def init():
    py_init()
    py_fontInit()

def quit():
    py_fontQuit()
    py_quit()

def pumpEvents():
    py_eventPump()
