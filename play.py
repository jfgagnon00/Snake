"""
Responsable du jeu en mode interactif
"""


import click
import os
import pygame

from ai.agents import AgentInteractive
from configs import configsCreate
from core import Delegate
from game import GameSimulation
from graphics import GraphicWindow, init as gfxInit, quit as gfxQuit


class InputManager():
    def __init__(self):
        self._keyDownDelegate = Delegate()
        self._quitDelegate = Delegate()
        self._anyKeyPressedDelegate = Delegate()

    @property
    def keyDownDelegate(self):
        return self._keyDownDelegate

    @property
    def quitDelegate(self):
        return self._quitDelegate

    @property
    def anyKeyPressedDelegate(self):
        return self._anyKeyPressedDelegate

    def update(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._quitDelegate()

            if e.type == pygame.KEYDOWN:
                self._keyDownDelegate(e.key)

            if e.type == pygame.KEYUP:
                self._anyKeyPressedDelegate()

class InteractiveApplication():
    """
    Coordonne la simulation de tel sorte qu'un utiisateur
    peut y jouer a l'aide du clavier
    """
    def __init__(self, configs):
        gfxInit()

        self._quit = False
        self._importantMessage = None

        gridShape = (configs.simulation.gridWidth, configs.simulation.gridHeight)
        self._window = GraphicWindow(gridShape, configs.graphics)

        self._agent = AgentInteractive()

        self._simulation = GameSimulation(configs.simulation)
        self._simulationFpsDivider = configs.graphics.simulationFpsDivider
        self._simulationCounter = 0

        self._inputManager = InputManager()
        self._lastAnyKeyPressedCallable = None

        self._updateDelegate = Delegate()
        self._lastUpdateCallable = None

        # configure les delegates "statiques"
        self._updateDelegate.register(self._inputManager.update)
        self._inputManager.quitDelegate.register(self._onQuit)
        self._inputManager.keyDownDelegate.register(self._agent.onKeyDown)

        self._simulation.outOfBoundsDelegate.register(self._onLose)
        self._simulation.collisionDelegate.register(self._onLose)
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.turnDelegate.register(self._onSnakeTurn)
        self._simulation.moveDelegate.register(self._onSnakeMove)

    def run(self):
        self._quit = False
        self._reset()

        # en mode interactif, l'utilisateur doit
        # peser sur une touche avant de demarrer
        # (laisse le temps d'observer)
        self._setAnyKeyPressedState(self._onStartSimulation, "Pesez une touche démarrer")

        while not self._quit:
            self._updateDelegate()
            self._window.render(self._importantMessage)
            self._window.flip()

        self._done()

    def _done(self):
        gfxQuit()

    def _reset(self):
        self._simulation.reset()
        self._window.update(self._simulation)

    def _update(self):
        self._simulationCounter -= 1
        if self._simulationCounter <= 0:
            self._simulationCounter = self._simulationFpsDivider
            action = self._agent.getAction(self._simulation._snake.direction)

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

    def _setUpdateState(self, newUpdateCallable):
        self._simulationCounter = 0

        if not self._lastUpdateCallable is None:
            self._updateDelegate.unregister(self._lastUpdateCallable)

        self._lastUpdateCallable = newUpdateCallable

        if not self._lastUpdateCallable is None:
            self._updateDelegate.register(self._lastUpdateCallable)

    def _onStartSimulation(self):
        self._setAnyKeyPressedState(None)
        self._setUpdateState(self._update)

    def _onResetSimulation(self):
        self._setAnyKeyPressedState(None)
        self._reset()
        self._setUpdateState(self._update)

    def _onLose(self):
        self._setAnyKeyPressedState(self._onResetSimulation, "LOSER! - Pesez une touche pour redémarrer")
        self._setUpdateState(None)

    def _onWin(self):
        self._setAnyKeyPressedState(self._onResetSimulation, "WINNER! - Pesez une touche pour redémarrer")
        self._setUpdateState(None)
        self._window.update(self._simulation)

    def _onSnakeEat(self):
        # TODO: play sound
        self._window.update(self._simulation)

    def _onSnakeTurn(self):
        # TODO: play sound
        pass

    def _onSnakeMove(self):
        self._window.update(self._simulation)

    def _onQuit(self):
        self._quit = True

@click.command()
@click.option("--windowSize",
              "-w",
              type=int,
              help="Taille de la fenêtre d'affichage.")
@click.option("--fpsDivider",
              "-fd",
              type=int,
              help="Taille de la fenêtre d'affichage.")
def main(windowsize, fpsdivider):
    configs = configsCreate("config_overrides.json")

    if not windowsize is None and windowsize > 0:
        configs.graphics.windowSize = windowsize

    if not fpsdivider is None and fpsdivider > 0:
        configs.graphics.simulationFpsDivider = fpsdivider

    InteractiveApplication(configs).run()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    main()
