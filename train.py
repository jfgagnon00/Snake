"""
Demarrer le jeu en mode interactif
"""


import os

from ai import OpenAIGymAdapter
from configs import createConfigs
from game import GameEnvironment


class TrainApplication():
    """
    """
    # https://blog.paperspace.com/creating-custom-environments-openai-gym/
    def __init__(self, configs):
        self._agent = None
        self._env = OpenAIGymAdapter(GameEnvironment(configs.environment),
                                     configs.environment,
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

    configs = createConfigs("config_overrides.json")

    TrainApplication(configs).run()
