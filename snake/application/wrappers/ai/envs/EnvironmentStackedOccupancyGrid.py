from collections import deque
from copy import deepcopy
from gymnasium import ObservationWrapper, spaces
from numpy import array, \
                  newaxis, \
                  repeat

class EnvironmentStackedOccupancyGrid(ObservationWrapper):
    def __init__(self, env, maxlen):
        ObservationWrapper.__init__(self, env)

        occupancy_grid = env.observation_space["occupancy_grid"]
        occupancy_grid = spaces.Box(
            low=repeat(occupancy_grid.low[newaxis, ...], maxlen, axis=0),
            high=repeat(occupancy_grid.high[newaxis, ...], maxlen, axis=0),
            dtype=occupancy_grid.dtype
        )

        self._maxlen = maxlen
        self._occupancy_stack = deque(maxlen=maxlen)
        self._occupancy_dtype = occupancy_grid.dtype
        self.observation_space = deepcopy(env.observation_space)
        self.observation_space["occupancy_grid"] = occupancy_grid

    def reset(self, *args, seed=None, options=None):
        observations, info = self.env.reset(*args, seed=seed, options=options)
        for i in range(self._maxlen):
            self._occupancy_stack.append( observations["occupancy_grid"] )
        observations["occupancy_grid"] = self._toNumpy()
        return observations, info

    def step(self, *args):
        observations, reward, terinated, truncated, info = self.env.step(*args)
        self._occupancy_stack.append( observations["occupancy_grid"] )
        observations["occupancy_grid"] = self._toNumpy()
        return observations, reward, terinated, truncated, info

    def observation(self, observations):
        return observations

    def _toNumpy(self):
        return array(self._occupancy_stack, dtype=self._occupancy_dtype)
