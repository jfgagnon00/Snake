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
        self._env = gym.make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)
        self._agent = self._createAgent(configs.train, configs.simulation)

    def run(self):
        for e in tqdm(range(self._episodes)):
            done = False
            state = self._env.reset()

            while not done:
                action = self._agent.getAction(state)

                # TODO: s'assurer que les observations ne pointent pas sur le meme object
                newState, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated

                # Render the game
                self._env.render()

        self._env.close()

    def _createAgent(self, trainConfig, simulatinConfig):
        # instantier un agent a partir d'un string
        # limiter a ai.agents pour le moment
        agent_class = getattr(agents, trainConfig.agent)
        return agent_class(trainConfig, simulatinConfig)

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
@click.option("--agent",
              "-a",
              type=str,
              help="Type de l'agent à utiliser.")
@click.option("--windowSize",
              "-w",
              type=int,
              help="Taille de la fenêtre d'affichage.")
@click.option("--renderFps",
              "-fps",
              type=int,
              help="Frame Par Seconde de l'affichage.")
def main(unattended, episodes, agent, windowsize, renderfps):
    configs = configsCreate("config_overrides.json")
    configs.train.unattended = unattended

    if not episodes is None and episodes > 0:
        configs.train.episodes = episodes

    if not agent is None and len(agent) > 0:
        configs.train.agent = agent

    if not windowsize is None and windowsize > 0:
        configs.graphics.windowSize = windowsize

    if not renderfps is None and renderfps > 0:
        configs.environment.renderFps = renderfps

    TrainApplication(configs).run()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    main()
