from numpy import int32 as np_int32

from copy import deepcopy
from gymnasium import Env, spaces
from snake.configs import Rewards
from snake.core import Delegate
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

        numActions = len(GameAction)

        self.action_space = spaces.Discrete(numActions)
        self.observation_space = spaces.Dict({
                "occupancy_grid": spaces.Box(low=0,
                                             high=255,
                                             shape=(1, simulationConfig.gridHeight, simulationConfig.gridWidth),
                                             dtype=np_int32),
                "head_direction": spaces.Box(low=-1,
                                             high=1,
                                             shape=(2,),
                                             dtype=int),
                "head_position": spaces.Box(low=0,
                                            high=simulationConfig.gridHeight - 1,
                                            shape=(2,),
                                            dtype=int),
                "food_position": spaces.Box(low=0,
                                            high=simulationConfig.gridHeight - 1,
                                            shape=(2,),
                                            dtype=int),
                "available_actions": spaces.Box(low=0,
                                                high=numActions - 1,
                                                shape=(numActions,),
                                                dtype=np_int32),
                "length": spaces.Discrete(simulationConfig.gridHeight * simulationConfig.gridWidth),
                "score": spaces.Discrete(simulationConfig.gridHeight * simulationConfig.gridWidth),
                "reward_type": spaces.Discrete(len(Rewards)),
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
        self._rewardType = Rewards.UNKNOWN
        self._rewardTypes = list(Rewards)
        self._done = False
        self._maxVisitCount = 2 if trainConfig is None else trainConfig.maxVisitCount

        self._trappedDelegate = Delegate()

        # configuer les delegates pour gerer les recompenses
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.moveDelegate.register(self._onSnakeMove)

    @property
    def trappedDelegate(self):
        return self._trappedDelegate

    @property
    def winDelegate(self):
        return self._simulation.winDelegate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._simulation.reset(options=options)
        self._maybeUpdateWindow(reset=True)

        return self._getObservations(), self._getInfo()

    def step(self, action):
        # reset recompense (les delegates vont le mettre la jour)
        self._rewardType = Rewards.UNKNOWN
        self._done = False

        # simulation:
        # les evenements appropries seront lances par les delegates et feront avancer les etats
        self._simulation.apply(action)

        # faire affichage si besoin
        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        # detection de boucle infinie: empecher serpent de passer plus de
        # self._maxVisitCount fois au meme endroit. Moins restrictif que
        # longueur d'episode et plus efficace
        truncated = self._simulation.occupancyGridCount.max() > self._maxVisitCount
        if truncated and self._maxVisitCount > 0:
            self._rewardType = Rewards.TRUNCATED

        observations = self._getObservations()

        # la simulation a besoin d'une action supplementaire dans certains cas pour
        # signaler les etats terminaux; en tenir compte immediatement pour faciliter
        # les etapes ulterieurs
        if observations["available_actions"].sum() == 0:
            self._done = True
            self._rewardType = Rewards.TRAPPED
            observations["reward_type"] = self._rewardTypes.index(self._rewardType)
            self._trappedDelegate()

        return observations, \
               self._rewards[self._rewardType], \
               self._done, \
               truncated, \
               self._getInfo()

    def render(self):
        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

    def close(self):
        if not self._window is None:
            gfxQuit()

    def _getObservations(self):
        observations = self._simulation.getObservations()
        observations["reward_type"] = self._rewardTypes.index(self._rewardType)
        return observations

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

    def _onSnakeEat(self):
        self._rewardType = Rewards.EAT
        self._maybeUpdateWindow()

    def _onSnakeMove(self):
        self._rewardType = Rewards.MOVE
        self._maybeUpdateWindow()

    def _onWin(self):
        self._done = True
        self._rewardType = Rewards.WIN
        self._maybeUpdateWindow()
