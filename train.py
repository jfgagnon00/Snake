"""
Responsable de l'entrainement des agents
"""


import ai
import ai.agents as agents
import click
import gymnasium as gym
import os

from configs import configsCreate
from tqdm import tqdm


class TrainApplication():
    def __init__(self, configs):
        self._episodes = configs.train.episodes
        self._agent = agents.AgentRandom()
        self._env = gym.make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)

    def run(self):
        for e in tqdm(range(self._episodes)):
            terminated = False
            observation = self._env.reset()

            while not terminated:
                action = self._agent.getAction(observation)

                # TODO: s'assurer que les observations ne pointent pas sur le meme object
                newObervation, reward, terminated, truncated, info = self._env.step(action)

                # Render the game
                self._env.render()

        self._env.close()

@click.command()
@click.option("--unattended",
              "-u",
              is_flag=True,
              default=False,
              help="Train without rendering.")
@click.option("--episodes",
              "-e",
              type=int,
              help="Episode count to train.")
def main(unattended, episodes):
    configs = configsCreate("config_overrides.json")
    configs.train.unattended = unattended

    if not episodes is None and episodes > 0:
        configs.train.episodes = episodes

    TrainApplication(configs).run()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    main()
