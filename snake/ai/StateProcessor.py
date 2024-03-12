import numpy as np

from collections import deque
from pprint import pprint
from snake.core import Vector
from snake.configs import Rewards
from snake.game import GameDirection, GridOccupancy


class _StateProcessor(object):
    """
    Encapsule la conversion des observations vers numpy arrays
    """

    def __call__(self, state, info):
        # grid = self._applySymmetry(state)
        # grid = state["occupancy_grid"].squeeze() / 255.0
        # grid = state["occupancy_grid"].squeeze()

        # showFood = info["reward_type"] > Rewards.EAT
        occupancyGrid = self._splitOccupancyGrid(state["occupancy_grid"], pad=False, showFood=True)
        distanceGrid = self._distanceGrid(state, info)

        return np.vstack((occupancyGrid, distanceGrid), dtype=np.float32), None
            #    info["available_actions"].copy()
            #    np.concatenate((state["head_direction"], info["available_actions"]))

    def _applySymmetry(self, state):
        # simplifier state: toujours mettre par rapport a NORTH
        k = self._rot90WithNorth(state)
        grid = state["occupancy_grid"]
        grid = np.rot90(grid, k=k, axes=(1, 2))

        return grid.copy()

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

    def _distanceGrid(self, state, info):
        stack = deque()

        occupancyGrid = state["occupancy_grid"]
        G = np.zeros_like(occupancyGrid, dtype=np.float32)

        food = info["food_position"]
        if not food is None:
            food = Vector.fromNumpy(food)

            G[0, food.y, food.x] = 1
            stack.append(food)

            while len(stack) > 0:
                p = stack.pop()
                r = -0.1 + 0.99 * G[0, p.y, p.x]
                r = max(r, 1e-4)

                for v in list(GameDirection):
                    q = p + v.value
                    if self._isEmpty(occupancyGrid, q) and G[0, q.y, q.x] == 0:
                        G[0, q.y, q.x] = r
                        stack.append(q)

        return G

    def _isEmpty(self, occupancyGrid, p):
        _, h, w = occupancyGrid.shape

        if p.x < 0 or p.x >= w:
            return False

        if p.y < 0 or p.y >= h:
            return False

        occupancy = occupancyGrid[0, p.y, p.x]

        return occupancy == GridOccupancy.EMPTY

