"""
Gere l'afficahge de la simulation
"""


import pygame

from .GraphicWindow import GraphicWindow


def init():
    pygame.init()
    pygame.font.init()

def quit():
    pygame.font.quit()
    pygame.quit()

def pumpEvents():
    pygame.event.pump()
