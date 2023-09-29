import pygame

from .Sprite import Sprite

class GraphicWindow():
    """
    Encapsule les objets pygame pour l'affichage et
    l'animation de GameEnvironment
    """
    def __init__(self, simulationGridShape, graphicsConfig):
        # attention: simulationGridShape utlise la
        # convention numpy => (h, w)
        aspectRatio = simulationGridShape[1] / simulationGridShape[0]
        if aspectRatio >= 1:
            w = int(graphicsConfig.windowSize)
            h = int(graphicsConfig.windowSize / aspectRatio)
            tileSize = w / simulationGridShape[1]
        else:
            w = int(graphicsConfig.windowSize * aspectRatio)
            h = int(graphicsConfig.windowSize)
            tileSize = h / simulationGridShape[0]

        self._window = pygame.display.set_mode((w, h))
        self._clock = pygame.time.Clock()
        self._fps = graphicsConfig.fps
        self._clearColor = graphicsConfig.clearColor
        self._initBackgroundTiles(tileSize,
                                  simulationGridShape,
                                  graphicsConfig)

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
        tileLight = pygame.surface.Surface((tileSize, tileSize))
        tileLight.fill(graphicsConfig.backgroundTileColorLight)

        tileDark = pygame.surface.Surface((tileSize, tileSize))
        tileDark.fill(graphicsConfig.backgroundTileColorDark)

        tileSurfaces = (tileLight, tileDark)
        rowTileSurfaceIndex = 0

        self._backgroundTiles = pygame.sprite.Group()

        for y in range(simulationGridShape[0]):
            tileSurfaceIndex = rowTileSurfaceIndex
            for x in range(simulationGridShape[1]):
                sprite = Sprite(tileSurfaces[tileSurfaceIndex])
                sprite.rect.x = x * tileSize
                sprite.rect.y = y * tileSize
                self._backgroundTiles.add(sprite)
                tileSurfaceIndex = 1 - tileSurfaceIndex
            rowTileSurfaceIndex = 1 - rowTileSurfaceIndex
