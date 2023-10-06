"""
Responsable de l'entrainement des agents
"""


import ai
import ai.agents as agents
import click
import gymnasium as gym
import os

from configs import configsCreate


class TrainApplication():
    def __init__(self, configs):
        self._agent = agents.AgentRandom()
        self._env = gym.make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)

    def run(self):
        observation = self._env.reset()

        while True:
            action = self._agent.getAction(observation)

            # TODO: s'assurer que les observations ne pointent pas sur le meme object
            newObervation, reward, terminated, truncated, info = self._env.step(action)

            # Render the game
            self._env.render()

            if terminated:
                break

        self._env.close()

@click.command()
@click.option("--unattended",
              "-u",
              is_flag=True,
              default=False,
              help="Train without rendering.")
def main(unattended):
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    configs = configsCreate("config_overrides.json")
    configs.train.unattended = unattended

    TrainApplication(configs).run()

if __name__ == "__main__":
    main()
