from __future__ import annotations

import numpy as np

from gym import Env, spaces
from game import GameAction
from game import GameSimulation
from game import GridOccupancy
from graphics import GraphicWindow, init as gfxInit, quit as gfxQuit


class SnakeEnvironment(Env):
    _HUMAN = "human"
    _FPS = "render_fps"

    metadata = {
        "render_modes": [_HUMAN],
        _FPS: 0
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
        self.observation_space = spaces.Box(GridOccupancy.EMPTY,
                                            GridOccupancy.FOOD,
                                            # TODO: valider convention (w, h)
                                            shape=(simulationConfig.gridHeight, simulationConfig.gridWidth),
                                            dtype=np.int8)
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

        return self._get_obs(), reward, done, self._get_info()

    def render(self):
        # rien a faire
        pass

    def close(self):
        if not self._window is None:
            gfxQuit()

    def _get_obs(self):
        # TODO
        return None

    def _get_info(self):
        return None

    def _renderInternal(self):
        self._window.render()
        self._window.flip()
