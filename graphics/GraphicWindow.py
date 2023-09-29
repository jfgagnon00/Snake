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
            tileSize = int(graphicsConfig.windowSize / simulationGridShape[1])
        else:
            tileSize = int(graphicsConfig.windowSize / simulationGridShape[0])
        self._initBackgroundTiles(tileSize,
                                  simulationGridShape,
                                  graphicsConfig)

        w = simulationGridShape[1] * tileSize
        h = simulationGridShape[0] * tileSize
        self._window = pygame.display.set_mode((w, h))

    def render(self, gameEnvironment):
        self._window.fill(self._clearColor)
        self._backgroundTiles.draw(self._window)

    def flip(self):
        pygame.display.flip()
        self._clock.tick(self._fps)

    def _initBackgroundTiles(self,
                             tileSize,
                             simulationGridShape,
                             graphicsConfig):
        self._backgroundTiles = pygame.sprite.Group()

        tileLight = pygame.surface.Surface((tileSize, tileSize))
        tileLight.fill(graphicsConfig.backgroundTileColorLight)
        tileDark = pygame.surface.Surface((tileSize, tileSize))
        tileDark.fill(graphicsConfig.backgroundTileColorDark)
        tileSurfaces = (tileLight, tileDark)

        rowTileSurfaceIndex = 0
        y = 0
        for _ in range(simulationGridShape[0]):
            tileSurfaceIndex = rowTileSurfaceIndex
            x = 0
            for _ in range(simulationGridShape[1]):
                sprite = Sprite(tileSurfaces[tileSurfaceIndex])
                sprite.rect.x = x
                sprite.rect.y = y

                self._backgroundTiles.add(sprite)

                tileSurfaceIndex = 1 - tileSurfaceIndex
                x += tileSize

            rowTileSurfaceIndex = 1 - rowTileSurfaceIndex
            y += tileSize
