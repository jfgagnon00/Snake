from .GameConfig import GameConfig

class Snake():
    """
    Represente le serpent
    """

    def __init__(self, x, y, direction):
        # tete est toujours le dernier element de la liste
        # la queue est toujours l'element 0
        self.head = GameConfig.point(x, y)
        self.bodyParts = [self.head,
                           GameConfig.point(self.head.x-1, self.head.y),
                           GameConfig.point(self.head.x-2, self.head.y)]
        #self._state = Alive, Dead
        self.direction = direction

    #@property.getter
    #def head(self):
    #    return self._bodyParts[-1]

    #pas sure de mettre les moves la ??

    def _move(self, movement, Direction):
        x = self.head.x
        y = self.head.y
        if movement == Direction.RIGHT:
            x += GameConfig.block_size
        elif movement == Direction.LEFT:
            x -= GameConfig.block_size
        elif movement == Direction.DOWN:
            y += GameConfig.block_size
        elif movement == Direction.UP:
            y -= GameConfig.block_size
            
        self.head = GameConfig.point(x, y)

     