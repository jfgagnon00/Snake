from collections import namedtuple

class GameConfig():
    MOVE = "move"
    EAT = "eat"
    COLLIDE = "collide"
    point = namedtuple('Point', 'x, y')

    def __init__(self):
        self.grid_width = 10
        self.grid_height = 10
        self.block_size = 20
        self.rewards = {
            GameConfig.MOVE: 0,
            GameConfig.EAT: 10,
            GameConfig.COLLIDE: -10,
        }