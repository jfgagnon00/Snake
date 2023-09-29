from pygame.image import load
from pygame.sprite import Sprite as pygame_Sprite
from pygame.transform import smoothscale


class Sprite(pygame_Sprite):
    def __init__(self, image=None, filename=None):
        super().__init__()

        if not image is  None:
            self.image = image
        elif not filename is None:
            self.image = load(filename).convert_alpha()

        self.rect = self.image.get_rect()

    def resize(self, size):
        self.image = smoothscale(self.image, size)

        x = self.rect.x
        y = self.rect.y
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

