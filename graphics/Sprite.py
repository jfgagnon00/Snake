from pygame.sprite import Sprite as pygame_Sprite


class Sprite(pygame_Sprite):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()