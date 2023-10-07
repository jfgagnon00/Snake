from pygame import Surface
from pygame.sprite import Group

from .Sprite import _Sprite


class _GraphicBackground():
    """
    Represente le damier du background
    """
    def __init__(self, graphicsConfig, gridShape, gridCellSize, offset):
        self._group = Group()

        tileShape = (gridCellSize, gridCellSize)

        tileLight = Surface(tileShape)
        tileLight.fill(graphicsConfig.backgroundTileColorLight)

        tileDark = Surface(tileShape)
        tileDark.fill(graphicsConfig.backgroundTileColorDark)

        tileSurfaces = (tileLight, tileDark)

        rowTileSurfaceIndex = 0
        y = 0
        for _ in range(gridShape[1]):
            tileSurfaceIndex = rowTileSurfaceIndex
            x = 0
            for _ in range(gridShape[0]):
                _sprite = _Sprite(image=tileSurfaces[tileSurfaceIndex])
                _sprite.rect.x = x
                _sprite.rect.y = y + offset

                self._group.add(_sprite)

                tileSurfaceIndex = 1 - tileSurfaceIndex
                x += gridCellSize

            rowTileSurfaceIndex = 1 - rowTileSurfaceIndex
            y += gridCellSize

    def render(self, image):
        self._group.draw(image)