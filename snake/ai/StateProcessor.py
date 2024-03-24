import numpy as np

from snake.core import Vector
from snake.game import GameDirection
from .GridOccupancy import GridOccupancy


class _StateProcessor(object):
    """
    Encapsule la conversion des observations vers numpy arrays
    """

    def __call__(self, state, info):
        # ne supporte pas frameStack pour le moment

        state = info["simulation_state"]
        grid = np.full(state.gridShape, GridOccupancy.EMPTY, dtype=np.int32)
        for p in state.snake.bodyParts:
            grid[p.y, p.x] = GridOccupancy.SNAKE_BODY
        grid[state.snake.head.y, state.snake.head.x] = GridOccupancy.SNAKE_HEAD
        grid[state.snake.tail.y, state.snake.tail.x] = GridOccupancy.SNAKE_TAIL
        grid[state.food.y, state.food.x] = GridOccupancy.FOOD
        grid = np.expand_dims(grid, 0)

        grid = self._applySymmetry(grid, state.snake.direction)
        grid = self._splitOccupancyGrid(grid, pad=False, showFood=True)

        return grid, None

    def _applySymmetry(self, grid, direction):
        # simplifier state: toujours mettre par rapport a NORTH
        k = self._rot90WithNorth(direction)
        return np.rot90(grid, k=k, axes=(1, 2))

    def _rot90WithNorth(self, head_d):
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
