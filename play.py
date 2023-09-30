"""
Demarrer le jeu en mode interactif
"""


import os
import pygame

from agents.InteractiveAgent import InteractiveAgent
from configs.configs import createConfigs
from game.GameEnvironmentTemp import GameEnvironmentTemp
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
        self._window = GraphicWindow((10, 15), configs.graphics)
        self._environement = GameEnvironmentTemp(configs.environment)
        self._agent = InteractiveAgent()
        self._updateFnc = None

    def run(self):
        self._quit = False

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

    def _handleEvents(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._quit = True

            if e.type == pygame.KEYUP:
                self._anyKeyPressed = True

    def _update(self):
        action = self._agent.getAction()
        done = self._environement.apply(action)

        if done:
            self._setUpdateState(self._resetBeforeRestart,
                                 "LOSER! - Pesez une touche pour redémarrer")
        else:
            self._window.update(self._environement)

    def _waitForAnyKey(self):
        if self._anyKeyPressed:
            self._setUpdateState(self._update)

    def _resetBeforeRestart(self):
        if self._anyKeyPressed:
            self._environement.reset()
            self._setUpdateState(self._update)

    def _setUpdateState(self, updateFnc, message=None):
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
