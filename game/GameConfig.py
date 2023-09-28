

class GameConfig():
    MOVE = "move"
    EAT = "eat"
    COLLIDE = "collide"


    def __init__(self):
        self.grid_width = 10
        self.grid_height = 10
        self.rewards = {
            GameConfig.MOVE: 0,
            GameConfig.EAT: 10,
            GameConfig.COLLIDE: -10,
        }