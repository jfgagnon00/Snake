import numpy as np

from snake.core import Delegate
from snake.game import GameSimulation
from snake.graphics import GraphicWindow, init as gfxInit, quit as gfxQuit

from .InputManager import _InputManager
from .wrappers.ai.agents import AgentInteractive


class ApplicationInteractive(object):
    """
    Coordonne la simulation de tel sorte qu'un utiisateur
    peut y jouer a l'aide du clavier.
    """
    def __init__(self, configs):
        gfxInit()

        self._episode = 0
        self._quit = False
        self._importantMessage = None

        gridShape = (configs.simulation.gridWidth, configs.simulation.gridHeight)
        self.window = GraphicWindow(gridShape, configs.graphics)

        self.agent = AgentInteractive()

        self._simulation = GameSimulation(configs.simulation)
        self._simulationFpsDivider = configs.graphics.simulationFpsDivider
        self._simulationCounter = 0

        self._inputManager = _InputManager()
        self._lastAnyKeyPressedCallable = None

        self._updateDelegate = Delegate()
        self._lastUpdateCallable = None

        # configure les delegates "statiques"
        self._updateDelegate.register(self._inputManager.update)
        self._inputManager.quitDelegate.register(self._onQuit)
        self._inputManager.keyDownDelegate.register(self.agent.onKeyDown)

        self._simulation.outOfBoundsDelegate.register(self._onLose)
        self._simulation.collisionDelegate.register(self._onLose)
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.turnDelegate.register(self._onSnakeTurn)
        self._simulation.moveDelegate.register(self._onSnakeMove)

    def setPlaybackAgent(self, agent, quit=False):
        self.agent = agent
        agent.endOfActionsDelegate.register(self._onEndOfActionsDelegate)
        if quit:
            agent.endOfActionsDelegate.register(self._onQuit)

    def runAttended(self):
        # en mode interactif, l'utilisateur doit
        # peser sur une touche avant de demarrer
        # (laisse le temps d'observer)
        self._setAnyKeyPressedState(self._onStartSimulation, "Pesez une touche démarrer")
        self._runInternal()

    def runUnattended(self):
        # run le plus vite possible
        self._simulationFpsDivider = 1

        # quitte des que simulation est en etat terminal
        self._simulation.outOfBoundsDelegate.register(self._onQuit)
        self._simulation.collisionDelegate.register(self._onQuit)
        self._simulation.winDelegate.register(self._onQuit)

        # s'assurer que la simulation demarre sans attendre
        self._onStartSimulation()

        # go!
        self._runInternal()

    def _runInternal(self):
        self._quit = False
        self._reset()
        self.window.render()

        while not self._quit:
            self._updateDelegate()
            self.window.render(self._importantMessage)
            self.window.flip()

        self._done()

    def _done(self):
        gfxQuit()

    def _reset(self):
        self.agent.reset()
        self._simulation.reset()
        self.window.reset()
        self.window.update(self._simulation)

    def _update(self):
        self._simulationCounter -= 1
        if self._simulationCounter <= 0:
            self._simulationCounter = self._simulationFpsDivider
            action = self.agent.getAction(self._simulation.getObservations())

            # la simulation va lancer les evenements appropries
            # ceux-ci vont faire avancer les etats
            self._simulation.apply(action)

    def _setAnyKeyPressedState(self, newAnyKeyPressedCallable, message=None):
        self._importantMessage = message

        if not self._lastAnyKeyPressedCallable is None:
            self._inputManager.anyKeyPressedDelegate.unregister(self._lastAnyKeyPressedCallable)

        self._lastAnyKeyPressedCallable = newAnyKeyPressedCallable

        if not self._lastAnyKeyPressedCallable is None:
            self._inputManager.anyKeyPressedDelegate.register(self._lastAnyKeyPressedCallable)

    def _setUpdateState(self, newUpdateCallable, newSimulationCounter=0):
        self._simulationCounter = newSimulationCounter

        if not self._lastUpdateCallable is None:
            self._updateDelegate.unregister(self._lastUpdateCallable)

        self._lastUpdateCallable = newUpdateCallable

        if not self._lastUpdateCallable is None:
            self._updateDelegate.register(self._lastUpdateCallable)

    def _onStartSimulation(self):
        self._setAnyKeyPressedState(None)
        self._setUpdateState(self._update)

    def _onResetSimulation(self):
        self._episode += 1
        self._setAnyKeyPressedState(None)
        self._reset()
        self._setUpdateState(self._update, self._simulationFpsDivider)

    def _onEndOfActionsDelegate(self):
        self.agent.onEpisodeDone(self._episode)
        self._setAnyKeyPressedState(self._onResetSimulation, "Actions épuisées! - Pesez une touche pour redémarrer")
        self._setUpdateState(None)

    def _onLose(self):
        self.agent.onEpisodeDone(self._episode)
        self._setAnyKeyPressedState(self._onResetSimulation, "LOSER! - Pesez une touche pour redémarrer")
        self._setUpdateState(None)

    def _onWin(self):
        self.agent.onEpisodeDone(self._episode)
        self._setAnyKeyPressedState(self._onResetSimulation, "WINNER! - Pesez une touche pour redémarrer")
        self._setUpdateState(None)
        self.window.update(self._simulation)

    def _onSnakeEat(self):
        # TODO: play sound
        self.window.update(self._simulation)

    def _onSnakeTurn(self):
        # TODO: play sound
        pass

    def _onSnakeMove(self):
        self.window.update(self._simulation)

    def _onQuit(self):
        self.agent.onEpisodeDone(self._episode)
        self._quit = True
