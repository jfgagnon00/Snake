import numpy as np

from gymnasium import Env, spaces
from game import GameAction
from game import GameSimulation
from graphics import GraphicWindow, init as gfxInit, quit as gfxQuit, pumpEvents


class SnakeEnvironment(Env):
    _HUMAN = "human"
    _FPS = "render_fps"
    _RENDER_MODES = "render_modes"

    metadata = {
        _RENDER_MODES: [_HUMAN],
        _FPS: 60
    }

    def __init__(self,
                 renderMode=None,
                 environmentConfig=None,
                 simulationConfig=None,
                 graphicsConfig=None):
        super().__init__()

        assert not environmentConfig is None
        assert not simulationConfig is None
        assert renderMode is None or renderMode in SnakeEnvironment.metadata[SnakeEnvironment._RENDER_MODES]

        if graphicsConfig is None or environmentConfig.renderFps <= 0:
            renderMode = None

        self.action_space = spaces.Discrete(int(GameAction.COUNT))
        self.observation_space = spaces.Dict({
                "occupancy_grid": spaces.Box(low=0,
                                             high=255,
                                             shape=(simulationConfig.gridWidth, simulationConfig.gridHeight, 1),
                                             dtype=np.uint8),
                "head_direction": spaces.Box(low=-1,
                                             high=1,
                                             shape=(2,),
                                             dtype=int),
                "head_position": spaces.Box(low=np.array([0, 0]),
                                            high=np.array([simulationConfig.gridWidth - 1, simulationConfig.gridHeight - 1]),
                                            shape=(2,),
                                            dtype=int),
                "food_position": spaces.Box(low=np.array([0, 0]),
                                            high=np.array([simulationConfig.gridWidth - 1, simulationConfig.gridHeight - 1]),
                                            shape=(2,),
                                            dtype=int),
            })

        self._renderMode = renderMode
        self._simulation = GameSimulation(simulationConfig)
        self._window = None

        if self._renderMode == SnakeEnvironment._HUMAN:
            gfxInit()

            # environment override le fps
            # on veut que l'entraiment soit rapide et non interactif
            graphicsConfig.fps = environmentConfig.renderFps
            SnakeEnvironment.metadata[SnakeEnvironment._FPS] = graphicsConfig.fps

            self._window = GraphicWindow((simulationConfig.gridWidth, simulationConfig.gridHeight),
                                         graphicsConfig)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._simulation.reset()

        if not self._window is None:
            self._window.update(self._simulation)

        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        return self._get_obs(), self._get_info()

    def step(self, action):
        done = self._simulation.apply(action)

        if not done and not self._window is None:
            self._window.update(self._simulation)

        if self._renderMode == SnakeEnvironment._HUMAN:
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
            "head_position": self._simulation.snake.head.to_numpy(),
            "food_position": self._simulation.food.to_numpy(),
        }

    def _get_info(self):
        return {}

    def _renderInternal(self):
        pumpEvents()
        self._window.render()
        self._window.flip()
