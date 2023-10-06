"""
Demarrer le jeu en mode interactif
"""


import os

from ai import OpenAIGymAdapter
from configs import configsCreate
from game import GameSimulation


class TrainApplication():
    """
    """
    # https://blog.paperspace.com/creating-custom-environments-openai-gym/
    # https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
    def __init__(self, configs):
        self._agent = None
        self._env = OpenAIGymAdapter(GameSimulation(configs.simulation),
                                     configs.simulation,
                                     configs.train)

    def run(self):
        obs = self._env.reset()

        while True:
            # Take a random action
            action = self._env.action_space.sample()
            state, reward, done, info = self._env.step(action)

            # Render the game
            self._env.render()

            if done == True:
                break

        self._env.close()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    configs = configsCreate("config_overrides.json")

    TrainApplication(configs).run()
