from .GraphicWindow import GraphicWindow

import pygame

def init():
    pygame.init()
    pygame.font.init()

def quit():
    pygame.font.quit()
    pygame.quit()

def pumpEvents():
    pygame.event.pump()
