import numpy as np

from copy import deepcopy
from gymnasium import Env, spaces
from snake.configs import Rewards
from snake.game import GameAction
from snake.game import GameSimulation
from snake.graphics import GraphicWindow, init as gfxInit, quit as gfxQuit, pumpEvents


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
                 graphicsConfig=None,
                 trainConfig=None):
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
                                             shape=(1, simulationConfig.gridHeight, simulationConfig.gridWidth),
                                             dtype=np.int32),
                "occupancy_heatmap": spaces.Box(low=0,
                                                high=65535,
                                                shape=(1, simulationConfig.gridHeight, simulationConfig.gridWidth),
                                                dtype=np.int32),
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
                "score": spaces.Discrete(simulationConfig.gridHeight * simulationConfig.gridWidth),
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
        self._maxVisitCount = 2 if trainConfig is None else trainConfig.maxVisitCount

        # configuer les delegates pour gerer les recompenses
        self._simulation.outOfBoundsDelegate.register(self._onSnakeOutOfBounds)
        self._simulation.collisionDelegate.register(self._onSnakeCollision)
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.moveDelegate.register(self._onSnakeMove)

    @property
    def outOfBoundsDelegate(self):
        return self._simulation.outOfBoundsDelegate

    @property
    def collisionDelegate(self):
        return self._simulation.collisionDelegate

    @property
    def winDelegate(self):
        return self._simulation.winDelegate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observations = options.get("observations", None) if options else None
        infos = options.get("infos", None) if options else None
        self._simulation.reset(observations, infos)
        self._maybeUpdateWindow(reset=True)

        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        return self._getObservations(), self._getInfo()

    def step(self, action):
        # reset recompense (les delegates vont le mettre la jour)
        self._reward = 0
        self._done = False

        # simulation:
        # les evenements appropries seront lances par les delegates et feront avancer les etats
        self._simulation.apply(action)

        # detection de boucle infinie: empecher serpent de passer plus de
        # self._maxVisitCount fois au meme endroit. Moins restrictif que
        # longueur d'episode et plus efficace
        truncated = self._simulation.occupancyGridCount.max() > self._maxVisitCount
        if truncated and self._maxVisitCount > 0:
            self._reward = self._rewards[Rewards.TRUNCATED]

        # faire affichage si besoin
        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        return self._getObservations(), \
               self._reward, \
               self._done, \
               truncated, \
               self._getInfo()

    def render(self):
        # rien a faire
        pass

    def close(self):
        if not self._window is None:
            gfxQuit()

    def _getObservations(self):
        return self._simulation.getObservations()

    def _getInfo(self):
        return self._simulation.getInfo()

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
        self._reward += self._rewards[Rewards.OUT_OF_BOUNDS]
        self._done = True

    def _onSnakeCollision(self):
        self._reward += self._rewards[Rewards.COLLISION]
        self._done = True

    def _onSnakeEat(self):
        self._reward += self._rewards[Rewards.EAT]
        self._maybeUpdateWindow()

    def _onWin(self):
        self._reward += self._rewards[Rewards.WIN]
        self._done = True
        self._maybeUpdateWindow()

    def _onSnakeMove(self):
        self._reward += self._rewards[Rewards.MOVE]
        self._maybeUpdateWindow()
