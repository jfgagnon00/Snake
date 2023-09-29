import math
import pygame

from .GraphicsConfig import GraphicsConfig
from game.GameEnvironment import GameEnvironment

class GraphicWindow():
    """
    Encapsule les objets pygame pour l'affichage et
    l'animation de GameEnvironment
    """
    def __init__(self, aspectRatio, graphicsConfig):
        self._snake = None
        self._food = None

        if aspectRatio >= 1:
            w = int(graphicsConfig.windowSize)
            h = int(graphicsConfig.windowSize / aspectRatio)
        else:
            w = int(graphicsConfig.windowSize * aspectRatio)
            h = int(graphicsConfig.windowSize)

        self._window = pygame.display.set_mode((w, h))
        self._clock = pygame.time.Clock()
        self._fps = graphicsConfig.fps
        self._clearColor = graphicsConfig.clearColor

    def render(self, gameEnvironment):
        self._window.fill(self._clearColor)
        pygame.display.flip()
        self._clock.tick(self._fps)