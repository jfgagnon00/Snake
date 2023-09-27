
import random
from .GameConfig import GameConfig

class Food():
      
      def __init__(self):
          self.food = None
#pas sure 

      def place_food(self):
          self.x = random.randint(0, (self.GameConfig.grid_width-GameConfig.block_size )//GameConfig.block_size )*GameConfig.block_size
          self.y = random.randint(0, (self.GameConfig.grid_height-GameConfig.block_size )//GameConfig.block_size )*GameConfig.block_size
          Food.food = GameConfig.point(self.x, self.y)
