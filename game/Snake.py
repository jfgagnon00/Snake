from .GameConfig import GameConfig

class Snake():
    """
    Represente le serpent
    """

    def __init__(self):
        # tete est toujours le dernier element de la liste
        # la queue est toujours l'element 0
        self.head = GameConfig.point(self.GameConfig.grid_width/2, self.GameConfig.grid_height/2)
        self.bodyParts = [self.head,
                           GameConfig.point(self.head.x-GameConfig.block_size, self.head.y),
                           GameConfig.point(self.head.x-(2*GameConfig.block_size), self.head.y)]
        #self._state = Alive, Dead
        self.direction = None

    #@property.getter
    #def head(self):
    #    return self._bodyParts[-1]

    #pas sure de mettre les moves la ??

    #def move(self, movement, direction): 
    #    def _move(self, direction):
    #    x = self.head.x
    #    y = self.head.y
    #    if direction == Direction.RIGHT:
    #        x += GameConfig.block_size
    #    elif direction == Direction.LEFT:
    #        x -= GameConfig.block_size
    #    elif direction == Direction.DOWN:
    #        y += GameConfig.block_size
    #    elif direction == Direction.UP:
    #        y -= GameConfig.block_size
            
    #    self.head = Point(x, y)

     