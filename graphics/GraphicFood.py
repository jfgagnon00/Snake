from pygame.sprite import Group

from .Sprite import _Sprite

class _GraphicFood():
    def __init__(self, graphicsConfig, size):
        self._sprite = _Sprite(filename=graphicsConfig.foodSpritePath)
        self._sprite.resize((size, size))
        self._sprite.optimize(True)
        self._group = Group(self._sprite)

    @property
    def rect(self):
        return self._sprite.rect

    def render(self, image):
        self._group.draw(image)
