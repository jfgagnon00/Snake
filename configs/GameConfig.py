

class GameConfig():
    MOVE = "move"
    EAT = "eat"
    COLLIDE = "collide"


    def __init__(self):
        self.gridWidth = 10
        self.gridHeight = 10
        self.rewards = {
            GameConfig.MOVE: 0,
            GameConfig.EAT: 10,
            GameConfig.COLLIDE: -10,
        }