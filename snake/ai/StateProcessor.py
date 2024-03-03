import numpy as np

from snake.core import Vector
from snake.game import GameDirection, GridOccupancy


class _StateProcessor(object):
    """
    Encapsule la conversion des observations vers numpy arrays
    """

    def __call__(self, state):
        grid, food_flags = self._applySymmetry(state)

        showFood = state["reward_type"] > 0
        grid = self._splitOccupancyGrid(grid, pad=False, showFood=showFood)

        if False:
            flags = np.array(food_flags)
        else:
            flags = None

        return grid, flags

    def _applySymmetry(self, state):
        # simplifier state: toujours mettre par rapport a NORTH
        k = self._rot90WithNorth(state)
        grid = state["occupancy_grid"]
        grid = np.rot90(grid, k=k, axes=(1, 2))

        return grid.copy(), \
               self._foodFlags(state)

    def _rot90WithNorth(self, state):
        head_d = state["head_direction"]
        head_d = Vector.fromNumpy(head_d)

        head_d = GameDirection(head_d)
        if head_d == GameDirection.EAST:
            # CCW
            return 1
        elif head_d == GameDirection.WEST:
            # CW
            return -1
        elif head_d == GameDirection.SOUTH:
            # 180 degrees
            return 2
        else:
            return 0

    def _rot90Vector(self, v, k):
        if not v is None:
            v  = Vector.fromNumpy(v)
            v -= self._gridCenter
            v  = v.rot90(k)
            v += self._gridCenter
            v  = v.toInt()
        return v

    def _foodFlags(self, state):
        food_cw = food_ccw = food_f = food_b = 0

        food_p = state["food_position"]
        if not food_p is None:
            food_p = Vector.fromNumpy(food_p)

            head_p = state["head_position"]
            head_p = Vector.fromNumpy(head_p)

            food_d = food_p - head_p

            head_d = state["head_direction"]
            head_d = Vector.fromNumpy(head_d)

            dot_ = Vector.dot(head_d, food_d)

            if dot_ > 0:
                food_f = 1

            if dot_ < 0:
                food_b = 1

            w = Vector.winding(head_d, food_d)

            if w == -1:
                food_cw = 1

            if w == 1:
                food_ccw = 1

        return food_cw, food_ccw, food_f, food_b

    def _splitOccupancyGrid(self, occupancyGrid, pad=False, showFood=True):
        if pad:
            # pad le pourtour pour avoir obstacle
            occupancyGrid = np.pad(occupancyGrid,
                                    ((0, 0), (1, 1), (1, 1)),
                                    constant_values=GridOccupancy.SNAKE_BODY)

        shape = (3, *occupancyGrid.shape[1:])

        occupancyStack = np.zeros(shape=shape, dtype=np.int32)
        occupancyStack[0] = np.where(occupancyGrid[0,:,:] == GridOccupancy.SNAKE_BODY, 1, 0)
        occupancyStack[1] = np.where(occupancyGrid[0,:,:] == GridOccupancy.SNAKE_HEAD, 1, 0)

        if showFood:
            occupancyStack[2] = np.where(occupancyGrid[0,:,:] == GridOccupancy.FOOD, 1, 0)

        return occupancyStack
