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

from wrappers.ai.agents import AgentActionRecorder, AgentActionPlayback
from wrappers.graphics import VideoWriter


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
    peut y jouer a l'aide du clavier.
    """
    def __init__(self, configs):
        gfxInit()

        self._quit = False
        self._importantMessage = None

        gridShape = (configs.simulation.gridWidth, configs.simulation.gridHeight)
        self.window = GraphicWindow(gridShape, configs.graphics)

        self.agent = AgentInteractive()

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
        self._inputManager.keyDownDelegate.register(self.agent.onKeyDown)

        self._simulation.outOfBoundsDelegate.register(self._onLose)
        self._simulation.collisionDelegate.register(self._onLose)
        self._simulation.eatDelegate.register(self._onSnakeEat)
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.turnDelegate.register(self._onSnakeTurn)
        self._simulation.moveDelegate.register(self._onSnakeMove)

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
        self.window.update(self._simulation)

    def _update(self):
        self._simulationCounter -= 1
        if self._simulationCounter <= 0:
            self._simulationCounter = self._simulationFpsDivider
            action = self.agent.getAction(self._simulation._snake.direction)

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
        self.agent.onSimulationDone()
        self._setAnyKeyPressedState(self._onResetSimulation, "LOSER! - Pesez une touche pour redémarrer")
        self._setUpdateState(None)

    def _onWin(self):
        self.agent.onSimulationDone()
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
@click.option("--record",
              type=str,
              help="Nom de fichier pour enregistrer des parties. Inclue le chemin. % sera remplacer par "
                   "le numéro de partie. Le format est toujours json. Ex: recordings/game_%.json")
@click.option("--playback",
              type=str,
              help="Nom de l'enregistrement a rejouer. Ex: recordings/game_%.json")
@click.option("--movie",
              is_flag=True,
              default=False,
              help="Enregistre le playback dans un fichier .mp4.")
def main(windowsize, fpsdivider, record, playback, movie):
    configs = configsCreate("config_overrides.json")

    if not windowsize is None and windowsize > 0:
        configs.graphics.windowSize = windowsize

    if not fpsdivider is None and fpsdivider > 0:
        configs.graphics.simulationFpsDivider = fpsdivider

    application = InteractiveApplication(configs)

    unattended = False
    if not record is None:
        # override agent pour gerer record
        application.agent = AgentActionRecorder(application.agent, record)
        application.window.caption += " - recording"
    elif not playback is None:
        # override agent pour gerer playback
        application.agent = AgentActionPlayback(playback)
        application.window.caption += " - playback"
        if movie:
            # override window pour avoir enregistrement video
            filename, _ = os.path.splitext(playback)
            filename = f"{filename}.mp4"
            fps = configs.graphics.fps / configs.graphics.simulationFpsDivider
            application.window = VideoWriter(application.window, fps, filename)
            unattended = True

    if unattended:
        application.runUnattended()
        application.window.dispose()
    else:
        application.runAttended()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    main()
