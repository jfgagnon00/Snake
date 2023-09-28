import pygame

from .GraphicsConfig import GraphicsConfig
from ..game.GameEnvironment import GameEnvironment

class GraphicWindow():
    """
    Encapsule les objets pygame pour l'affichage de GameEnvironment
    """
    def __init__(self, graphicsConfig):
        self._snake = None
        self._food = None

        self._window = None

    def render(self, gameEnvironment):
        pass