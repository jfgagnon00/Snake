import numpy as np

from configs import Rewards
from copy import deepcopy
from gymnasium import Env, spaces
from game import GameAction
from game import GameSimulation
from graphics import GraphicWindow, init as gfxInit, quit as gfxQuit, pumpEvents


class SnakeEnvironment(Env):
    """
    Responsable de faire le pont entre le jeu de snake et OpenAI Gymnasium.
    Parametres de configuration passent par les objets du module configs.
    """
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

        self.action_space = spaces.Discrete(len(GameAction))
        self.observation_space = spaces.Dict({
                "occupancy_grid": spaces.Box(low=0,
                                             high=255,
                                             shape=(simulationConfig.gridHeight, simulationConfig.gridWidth, 1),
                                             dtype=np.uint8),
                "head_direction": spaces.Box(low=-1,
                                             high=1,
                                             shape=(2,),
                                             dtype=int),
                "head_position": spaces.Box(low=np.array([0, 0]),
                                            high=np.array([simulationConfig.gridHeight - 1, simulationConfig.gridWidth - 1]),
                                            shape=(2,),
                                            dtype=int),
                "food_position": spaces.Box(low=np.array([0, 0]),
                                            high=np.array([simulationConfig.gridHeight - 1, simulationConfig.gridWidth - 1]),
                                            shape=(2,),
                                            dtype=int),
                "length": spaces.Discrete(simulationConfig.gridHeight * simulationConfig.gridWidth),
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

        self._rewards = deepcopy(environmentConfig.rewards)
        self._reward = 0
        self._done = False

        # configuer les delegates pour gerer les recompenses
        self._simulation.outOfBoundsDelegate.register(self._onSnakeOutOfBounds)
        self._simulation.collisionDelegate.register(self._onSnakeCollision)
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.moveDelegate.register(self._onSnakeMove)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._simulation.reset()
        self._maybeUpdateWindow(reset=True)

        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        return self._getObservaton(), self._getInfo()

    def step(self, action):
        # reset recompense (les delegates vont le mettre la jour)
        self._reward = 0
        self._done = False

        # simulation:
        # les evenements appropries seront lances par les delegates et feront avancer les etats
        self._simulation.apply(action)

        # faire affichage si besoin
        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        return self._getObservaton(), self._reward, self._done, False, self._getInfo()

    def render(self):
        # rien a faire
        pass

    def close(self):
        if not self._window is None:
            gfxQuit()

    def _getObservaton(self):
        return {
            "occupancy_grid": np.expand_dims(self._simulation.occupancyGrid, axis=-1).copy(),
            "head_direction": self._simulation.snake.direction.to_numpy(),
            "head_position": self._simulation.snake.head.to_numpy(),
            "food_position": self._simulation.food.to_numpy(),
            "length": self._simulation.snake.length,
        }

    def _getInfo(self):
        return {}

    def _renderInternal(self):
        pumpEvents()
        self._window.render()
        self._window.flip()

    def _maybeUpdateWindow(self, reset=False):
        if not self._window is None:
            if reset:
                self._window.reset()
            self._window.update(self._simulation)

    def _onSnakeOutOfBounds(self):
        self._reward = self._rewards[Rewards.OUT_OF_BOUNDS]
        self._done = True

    def _onSnakeCollision(self):
        self._reward = self._rewards[Rewards.COLLISION]
        self._done = True

    def _onSnakeEat(self):
        self._reward = self._rewards[Rewards.EAT]
        self._maybeUpdateWindow()

    def _onWin(self):
        self._reward = self._rewards[Rewards.WIN]
        self._done = True
        self._maybeUpdateWindow()

    def _onSnakeMove(self):
        self._reward = self._rewards[Rewards.MOVE]
        self._maybeUpdateWindow()
