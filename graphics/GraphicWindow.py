import pygame

from .Sprite import Sprite

class GraphicWindow():
    """
    Encapsule les objets pygame pour l'affichage et
    l'animation de GameEnvironment
    """
    def __init__(self, simulationGridShape, graphicsConfig):
        self._clock = pygame.time.Clock()
        self._fps = graphicsConfig.fps
        self._clearColor = graphicsConfig.clearColor

        # attention #1: simulationGridShape utlise la convention numpy => (h, w)
        # attention #2: pygame aime bien les coordonnes en pixels, attention aux
        #               operations en nombres flotant
        aspectRatio = simulationGridShape[1] / simulationGridShape[0]
        if aspectRatio >= 1:
            self._tileSize = int(graphicsConfig.windowSize / simulationGridShape[1])
        else:
            self._tileSize = int(graphicsConfig.windowSize / simulationGridShape[0])
        self._initBackgroundTiles(simulationGridShape, graphicsConfig)

        w = simulationGridShape[1] * self._tileSize
        h = simulationGridShape[0] * self._tileSize
        self._canvas = pygame.display.set_mode((w, h))

        self._initFood(graphicsConfig)

    def update(gameEnvironment):
        pass

    def render(self, ):
        self._canvas.fill(self._clearColor)
        self._backgroundTiles.draw(self._canvas)
        self._food.draw(self._canvas)

    def flip(self):
        pygame.display.flip()
        self._clock.tick(self._fps)

    def _initBackgroundTiles(self, simulationGridShape, graphicsConfig):
        self._backgroundTiles = pygame.sprite.Group()

        tileShape = (self._tileSize, self._tileSize)
        tileLight = pygame.surface.Surface(tileShape)
        tileLight.fill(graphicsConfig.backgroundTileColorLight)
        tileDark = pygame.surface.Surface(tileShape)
        tileDark.fill(graphicsConfig.backgroundTileColorDark)
        tileSurfaces = (tileLight, tileDark)

        rowTileSurfaceIndex = 0
        y = 0
        for _ in range(simulationGridShape[0]):
            tileSurfaceIndex = rowTileSurfaceIndex
            x = 0
            for _ in range(simulationGridShape[1]):
                sprite = Sprite(image=tileSurfaces[tileSurfaceIndex])
                sprite.rect.x = x
                sprite.rect.y = y

                self._backgroundTiles.add(sprite)

                tileSurfaceIndex = 1 - tileSurfaceIndex
                x += self._tileSize

            rowTileSurfaceIndex = 1 - rowTileSurfaceIndex
            y += self._tileSize

    def _initFood(self, graphicsConfig):
        self._foodSprite = Sprite(filename=graphicsConfig.foodSpritePath)
        self._foodSprite.resize((self._tileSize, self._tileSize))
        self._food = pygame.sprite.Group(self._foodSprite)

