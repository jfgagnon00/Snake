from pygame.image import load
from pygame.rect import Rect
from pygame.sprite import Sprite as pygame_Sprite
from pygame.transform import smoothscale


class Sprite(pygame_Sprite):
    def __init__(self, image=None, filename=None):
        super().__init__()

        if not image is  None:
            self.image = image
        elif not filename is None:
            self.image = load(filename)
        else:
            self.image = None

        if self.image is None:
            self.rect = Rect()
        else:
            self.rect = self.image.get_rect()

    def resize(self, size):
        self.image = smoothscale(self.image, size)

        x = self.rect.x
        y = self.rect.y
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def optimize(self, alpha=False):
        if alpha:
            self.image = self.image.convert_alpha()
        else:
            self.image = self.image.convert()

