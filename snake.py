"""
'Colle' pour demarrer le jeu dans les divers modes possibles
"""
from game.GameConfig import GameConfig
from game.GameEnvironment import GameEnvironment
from misc.MetaObject import MetaObject

gameConfig = GameConfig()


if __name__ == "__main__":
    print("Init")
    env = GameEnvironment(gameConfig)

    print("Test forward")
    # env.apply(GameEnvironment.MOVEMENT_FORWARD)
    
    print("Test reset")
    # env.reset()


