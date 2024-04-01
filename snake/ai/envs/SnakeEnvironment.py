from numpy import int32 as np_int32

from copy import deepcopy
from gymnasium import Env, spaces
from snake.configs import Rewards
from snake.core import Delegate
from snake.game import GameAction, GameSimulation, GameSimulationState
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
                "dummy": spaces.Discrete(1),
            })

        self._renderMode = renderMode
        self._simulation = GameSimulation()
        self._simulationState = GameSimulationState(simulationConfig)
        self._window = None

        if self._renderMode == SnakeEnvironment._HUMAN:
            gfxInit()

            # environment override le fps
            # on veut que l'entraiment soit rapide et non interactif
            graphicsConfig.fps = environmentConfig.renderFps
            SnakeEnvironment.metadata[SnakeEnvironment._FPS] = graphicsConfig.fps

            self._window = GraphicWindow((simulationConfig.gridWidth, simulationConfig.gridHeight),
                                         graphicsConfig)

        # valider que les fichiers de configs ont bien
        # override les valeurs par defauts
        assert len(environmentConfig.rewards) == len(Rewards)

        self._rewards = deepcopy(environmentConfig.rewards)
        self._rewardType = Rewards.UNKNOWN
        self._done = False
        self._stepsWithoutFood = 0
        if trainConfig is None:
            self._maxStepsWithoutFood = 2 * simulationConfig.gridWidth * simulationConfig.gridHeight
        else:
            self._maxStepsWithoutFood = trainConfig.maxStepsWithoutFood

        self._trappedDelegate = Delegate()

        # configuer les delegates pour gerer les recompenses
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.loseDelegate.register(self._onLose)
        self._simulation.moveDelegate.register(self._onSnakeMove)

    @property
    def trappedDelegate(self):
        return self._trappedDelegate

    @property
    def winDelegate(self):
        return self._simulation.winDelegate

    @property
    def loseDelegate(self):
        return self._simulation.loseDelegate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._rewardType = Rewards.UNKNOWN
        self._done = False
        self._stepsWithoutFood = 0
        GameSimulationState.initRandom(self._simulationState)
        self._maybeUpdateWindow(reset=True)

        return self._getObservations(), self._getInfos()

    def step(self, action):
        # reset recompense (les delegates vont le mettre la jour)
        self._rewardType = Rewards.UNKNOWN
        self._done = False
        self._stepsWithoutFood += 1

        # simulation:
        # les evenements appropries seront lances par les delegates et feront avancer les etats
        self._simulation.apply(action, self._simulationState, inplace=True)

        # faire affichage si besoin
        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

        # detection de boucle infinie: empecher serpent de passer plus de
        # self._maxStepsWithoutFood fois au meme endroit. Moins restrictif que
        # longueur d'episode
        truncated = self._stepsWithoutFood > self._maxStepsWithoutFood
        if truncated and self._maxStepsWithoutFood > 0:
            self._rewardType = Rewards.TRUNCATED

        # la simulation a besoin d'une action supplementaire dans certains cas pour
        # signaler les etats terminaux; en tenir compte immediatement pour faciliter
        # les etapes ulterieures
        availableActions = self._simulationState.availableActions()
        if availableActions.sum() == 0:
            self._done = True
            self._rewardType = Rewards.TRAPPED
            self._trappedDelegate()

        return self._getObservations(), \
               self._rewards[self._rewardType.name], \
               self._done, \
               truncated, \
               self._getInfos(availableActions=availableActions)

    def render(self):
        if self._renderMode == SnakeEnvironment._HUMAN:
            self._renderInternal()

    def close(self):
        if not self._window is None:
            gfxQuit()

    def _getObservations(self):
        return self._simulation.getObservations(self._simulationState)

    def _getInfos(self, availableActions=None):
        infos = self._simulation.getInfos(self._simulationState)
        infos["reward_type"] = self._rewardType
        infos["available_actions"] = self._simulationState.availableActions() if availableActions is None else availableActions
        return infos

    def _renderInternal(self):
        pumpEvents()
        self._window.render()
        self._window.flip()

    def _maybeUpdateWindow(self, reset=False):
        if not self._window is None:
            if reset:
                self._window.reset()
            self._window.update(self._simulationState)

    def _onSnakeEat(self):
        self._rewardType = Rewards.EAT
        self._stepsWithoutFood = 0
        self._maybeUpdateWindow()

    def _onSnakeMove(self):
        self._rewardType = Rewards.MOVE
        self._maybeUpdateWindow()

    def _onWin(self):
        self._done = True
        self._rewardType = Rewards.WIN
        self._maybeUpdateWindow()

    def _onLose(self):
        self._done = True
        self._rewardType = Rewards.WIN
        self._maybeUpdateWindow()
