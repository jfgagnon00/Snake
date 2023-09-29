"""
Demarrer le jeu en mode interactif
"""


import os
import pygame

from configs.configs import createConfigs
from graphics.GraphicWindow import GraphicWindow


class InteractiveApplication():
    """
    Coordonne la simulation de tel sorte qu'un utiisateur
    peut y jouer a l'aide du clavier
    """
    def __init__(self, configs) -> None:
        pygame.init()

        self._quit = False
        self._environment = None
        self._window = GraphicWindow(1.0, configs.graphics)

    def run(self):
        while not self._quit:
            self._handleEvents()
            self._window.render(self._environment)

        self._done()

    def _done(self):
        pygame.quit()

    def _handleEvents(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._quit = True


if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    configs = createConfigs("config_overrides.json")

    InteractiveApplication(configs).run()
