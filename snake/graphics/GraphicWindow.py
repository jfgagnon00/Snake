from pygame import HIDDEN, SHOWN, SCALED
from pygame.font import Font
from pygame.display import set_mode, set_caption, get_caption, set_icon, flip
from pygame.time import Clock

from snake.core import Vector
from .Sprite import _Sprite
from .GraphicBackground import _GraphicBackground
from .GraphicFood import _GraphicFood
from .GraphicSnake import _GraphicSnake


class GraphicWindow(object):
    """
    Encapsule les objets pygame pour l'affichage et
    l'animation de GameSimulation
    """
    def __init__(self, simulationGridShape, graphicsConfig):
        self._clock = Clock()
        self._fps = graphicsConfig.fps
        self._clearColor = graphicsConfig.clearColor
        self._message = None
        self._frame = -1

        self._font = Font(graphicsConfig.fontPath,
                          size=graphicsConfig.fontSize)
        _, fontHeight = self._font.size("DEFAULT")
        self._fontMargin = int(fontHeight / 4)
        self._fontColor = graphicsConfig.fontColor
        self._gameAreaStart = fontHeight + 2 * self._fontMargin

        gameAreaSize = graphicsConfig.windowSize - self._gameAreaStart

        # attention #1: simulationGridShape utlise la convention (w, h)
        # attention #2: pygame aime bien les coordonnes en pixels, attention aux
        #               operations en nombres flotant
        aspectRatio = simulationGridShape[0] / simulationGridShape[1]
        if aspectRatio >= 1:
            self._tileSize = int(gameAreaSize / simulationGridShape[0])
        else:
            self._tileSize = int(gameAreaSize / simulationGridShape[1])

        w = simulationGridShape[0] * self._tileSize
        h = simulationGridShape[1] * self._tileSize + self._gameAreaStart

        icon = _Sprite(filename=graphicsConfig.iconPath)
        set_icon(icon.image)
        set_caption(graphicsConfig.caption)

        flags = SHOWN | SCALED if graphicsConfig.showWindow else HIDDEN
        self._window = set_mode((w, h), flags=flags, vsync=0)

        self._initBackground(graphicsConfig, simulationGridShape)
        self._initFood(graphicsConfig)
        self._initSnake(graphicsConfig)

    @property
    def caption(self):
        return get_caption()[0]

    @caption.setter
    def caption(self, value):
        set_caption(value)

    @property
    def fps(self):
        return self._fps

    @property
    def size(self):
        return self._window.get_rect().size

    @property
    def surface(self):
        return self._window

    def reset(self):
        self._frame = -1

    def update(self, gameEnvironment):
        self._frame += 1
        fps = int(self._clock.get_fps())
        score = gameEnvironment.score
        self._message = f"Score: {score:03d} FPS: {fps:2d} Frame: {self._frame}"

        if gameEnvironment.food is None:
            self._food.enable = False
        else:
            food = self._environmentToWindow(gameEnvironment.food)
            self._food.rect.x = food.x
            self._food.rect.y = food.y
            self._food.enable = True

        self._snake.update(gameEnvironment.snake)

    def render(self, message=None):
        self._window.fill(self._clearColor)
        self._background.render(self._window)
        self._food.render(self._window)
        self._snake.render(self._window)

        if message is None:
            message = self._message

        if not message is None:
            textImage = self._font.render(message, True, self._fontColor)
            self._window.blit(textImage, (self._fontMargin, self._fontMargin))

    def flip(self):
        flip()
        self._clock.tick(self._fps)

    def _initBackground(self, graphicsConfig, simulationGridShape):
        self._background = _GraphicBackground(graphicsConfig,
                                              simulationGridShape,
                                              self._tileSize,
                                              self._gameAreaStart)

    def _initFood(self, graphicsConfig):
        self._food = _GraphicFood(graphicsConfig, self._tileSize)
        self._food.rect.y += self._gameAreaStart

    def _initSnake(self, graphicsConfig):
        self._snake = _GraphicSnake(graphicsConfig,
                                    self._tileSize,
                                    self._gameAreaStart)

    def _environmentToWindow(self, vector):
        return Vector(vector.x * self._tileSize,
                      vector.y * self._tileSize + self._gameAreaStart)
