"""
Demarrer le jeu en mode interactif
"""


import os
import pygame

from agents.InteractiveAgent import InteractiveAgent
from configs.configs import createConfigs
from game.GameEnvironment import GameEnvironment
from graphics.GraphicWindow import GraphicWindow


class InteractiveApplication():
    """
    Coordonne la simulation de tel sorte qu'un utiisateur
    peut y jouer a l'aide du clavier
    """
    def __init__(self, configs):
        pygame.init()
        pygame.font.init()

        self._quit = False
        self._anyKeyPressed = False
        self._importantMessage = None

        gridShape = (configs.environment.grid_width, configs.environment.grid_height)
        self._window = GraphicWindow(gridShape, configs.graphics)

        self._environement = GameEnvironment(configs.environment)
        self._agent = InteractiveAgent()
        self._updateFnc = None
        self._simulationFpsDivider = configs.graphics.simulationFpsDivider
        self._simulationCounter = 0

    def run(self):
        self._quit = False
        self._reset()

        # en mode interactif, l'utilisateur doit
        # peser sur une touche avant de demarrer
        # (laisse le temps d'observer)
        self._setUpdateState(self._waitForAnyKey,
                             "Pesez une touche démarrer")

        while not self._quit:
            self._handleEvents()

            if not self._updateFnc is None:
                self._updateFnc()

            self._window.render(self._importantMessage)
            self._window.flip()

        self._done()

    def _done(self):
        pygame.font.quit()
        pygame.quit()

    def _reset(self):
        self._agent.reset()
        self._environement.reset()
        self._window.update(self._environement)

    def _handleEvents(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._quit = True

            if e.type == pygame.KEYDOWN:
                self._agent.onKeyDown(e.key)

            if e.type == pygame.KEYUP:
                self._anyKeyPressed = True

    def _update(self):
        self._simulationCounter -= 1
        if self._simulationCounter <= 0:
            self._simulationCounter = self._simulationFpsDivider

            action = self._agent.getAction(self._environement._snake.direction)
            if self._environement.apply(action):
                self._setUpdateState(self._resetBeforeRestart,
                                    "LOSER! - Pesez une touche pour redémarrer")
            else:
                self._window.update(self._environement)

    def _waitForAnyKey(self):
        if self._anyKeyPressed:
            self._setUpdateState(self._update)

    def _resetBeforeRestart(self):
        if self._anyKeyPressed:
            self._reset()
            self._setUpdateState(self._update)

    def _setUpdateState(self, updateFnc, message=None):
        self._simulationCounter = 0
        self._anyKeyPressed = False
        self._importantMessage = message
        self._updateFnc = updateFnc


if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    configs = createConfigs("config_overrides.json")

    InteractiveApplication(configs).run()
