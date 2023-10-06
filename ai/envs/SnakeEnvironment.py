import numpy as np

from gymnasium import Env, spaces
from game import GameAction
from game import GameSimulation
from game import GridOccupancy
from graphics import GraphicWindow, init as gfxInit, quit as gfxQuit


class SnakeEnvironment(Env):
    _HUMAN = "human"
    _FPS = "render_fps"

    metadata = {
        "render_modes": [_HUMAN],
        _FPS: 1000
    }

    def __init__(self,
                 render_mode=None,
                 environmentConfig=None,
                 simulationConfig=None,
                 graphicsConfig=None):
        super().__init__()

        assert not environmentConfig is None
        assert not simulationConfig is None
        assert render_mode is None or render_mode in SnakeEnvironment.metadata

        if graphicsConfig is None or environmentConfig.renderFps <= 0:
            render_mode = None

        self.action_space = spaces.Discrete(int(GameAction.COUNT))
        self.observation_space = spaces.Dict({
                "occupancy_grid": spaces.Box(low=0,
                                             high=255,
                                             shape=(simulationConfig.gridWidth, simulationConfig.gridHeight, 1),
                                             dtype=np.uint8),
                "head_direction": spaces.Box(low=0,
                                             high=1,
                                             shape=(2,),
                                             dtype=int),
                # "head_position": spaces.Box(low=[0, simulationConfig.gridWidth - 1],
                #                             high=[0, simulationConfig.gridHeight - 1],
                #                             shape=(2, 1),
                #                             dtype=int),
                # "food_position": spaces.Box(low=[0, simulationConfig.gridWidth - 1],
                #                             high=[0, simulationConfig.gridHeight - 1],
                #                             shape=(2, 1),
                #                             dtype=int),
            })

        self.render_mode = render_mode
        self._simulation = GameSimulation(simulationConfig)
        self._window = None

        if render_mode == SnakeEnvironment._HUMAN:
            gfxInit

            # environment override le fps, on veut que l'entraiment soit rapide et non interactif
            graphicsConfig.fps = environmentConfig.environmentConfig.renderFps

            SnakeEnvironment.metadata[SnakeEnvironment._FPS] = graphicsConfig._fps
            self._window = GraphicWindow((simulationConfig.gridWidth, simulationConfig.gridHeight),
                                         graphicsConfig)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._simulation.reset()

        if not self._window is None:
            self._simulation.reset(self._simulation)

        if self.render_mode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        return self._get_obs(), self._get_info()

    def step(self, action):
        done = self._simulation.apply(action)

        if not done and not self._window is None:
            self._window.update(self._simulation)

        if self.render_mode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        # TODO
        reward = 0

        return self._get_obs(), reward, done, False, self._get_info()

    def render(self):
        # rien a faire
        pass

    def close(self):
        if not self._window is None:
            gfxQuit()

    def _get_obs(self):
        return {
            "occupancy_grid": np.expand_dims(self._simulation.grid, axis=-1),
            "head_direction": self._simulation.snake.direction.to_numpy(),
            # "head_position": self._simulation.snake.head,
            # "food_position": self._simulation.food,
        }

    def _get_info(self):
        return {}

    def _renderInternal(self):
        self._window.render()
        self._window.flip()
