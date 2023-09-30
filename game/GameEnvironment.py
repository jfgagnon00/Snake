import numpy as np
import random
from enum import Enum
from game.GameConfig import GameConfig
from game.Snake import Snake
from game.Food import Food


class Direction(Enum):  # temp
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class GameEnvironment():
    """
    Responsable d'appliquer le mouvement au serpent
    """

    def __init__(self, gameConfig):
        #self._grid = []  # array 2d ???? GameConfig.grid_width, GameConfig.grid_height
        self._snake = Snake(4, 1, Direction.RIGHT)
        #self._food = Food()
        #self._score = 0
        #self._rewards = 0

    def reset(self):
        """
        Remet l'environment dans un etat initial
        """

        self._snake = Snake(4, 1, Direction.RIGHT)


        #self.score = 0
        #self.place_food()

    def place_food(self):
          self.x = random.randint(0, (self.GameConfig.grid_width-GameConfig.block_size )//GameConfig.block_size )*GameConfig.block_size
          self.y = random.randint(0, (self.GameConfig.grid_height-GameConfig.block_size )//GameConfig.block_size )*GameConfig.block_size
          Food.position = GameConfig.point(self.x, self.y)    

    def _move(self, direction):
        x = self._snake.head.x
        y = self._snake.head.y
        if direction == Direction.RIGHT:
            x += GameConfig.block_size
        elif direction == Direction.LEFT:
            x -= GameConfig.block_size
        elif direction == Direction.DOWN:
            y += GameConfig.block_size
        elif direction == Direction.UP:
            y -= GameConfig.block_size

        self.head = GameConfig.point(x, y)

    def _move(self, action):

        # Actions : [1, 0, 0] -> straight ; [0, 1, 0] -> right turn ; [0, 0, 1] -> left turn
        #[straight, right, left]
        self.direction = GameEnvironment.MOVEMENT_TURN_RIGHT
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # (pour revenir au debut du clock_wise)
            # right-turn right -> down -> left -> up
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # (pour revenir au debut du clock_wise)
            # left-turn right -> up -> left -> down
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self._snake.head.x
        y = self._snake.head.y
        if self.direction == GameEnvironment.Direction.RIGHT:
            x += GameConfig.block_size
        elif self.direction == GameEnvironment.Direction.LEFT:
            x -= GameConfig.block_size
        elif self.direction == GameEnvironment.Direction.DOWN:
            y += GameConfig.block_size
        elif self.direction == GameEnvironment.Direction.UP:
            y -= GameConfig.block_size

        self._snake.head = GameConfig.point(x, y)

    def apply(self, movement):
        """
        Applique le movement au serpent et met a jour les etats internes
        """
        # 1. bouge serpent
        self._snake._move(movement, Direction)  # update the head
        self._snake.insert(0, self._snake.head)

        # 2. mettre a jour grid

        # 3. resoudre collision
    def is_collision(self, pt=None):
        if pt is None:
            pt = self._snake.head
        # hits boundary
        if pt.x > self.w - GameConfig.block_size or pt.x < 0 or pt.y > self.h - GameConfig.point or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

        # 4. calculer reward et game over
        reward = 0
        game_over = False
        if self.is_collision():  # or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
        return reward, game_over, self.score
