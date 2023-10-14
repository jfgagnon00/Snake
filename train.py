"""
Responsable de l'entrainement des agents
"""


import ai # importe l'environnement "snake/SnakeEnvironment-v0"
import ai.agents as agents
import click
import gymnasium as gym
import os

from configs import configsCreate
from tqdm import tqdm
from wrappers.ai.agents import AgentActionRecorder


class TrainApplication():
    def __init__(self, configs):
        self._episodes = configs.train.episodes
        self.agent = self._createAgent(configs.train, configs.simulation)
        self._env = gym.make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)

    def run(self):
        for e in tqdm(range(self._episodes)):
            done = False
            state = self._env.reset()

            while not done:
                action = self.agent.getAction(state)

                # TODO: s'assurer que les observations ne pointent pas sur le meme object
                newState, reward, terminated, truncated, info = self._env.step(action)
                done = terminated or truncated

                # Render the game
                self._env.render()

            self.agent.onSimulationDone()

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
@click.option("--record",
              "-r",
              type=str,
              help="Nom de fichier pour enregistrer les épisodes. Inclue le chemin. % sera remplacer par "
                   "le numéro d'épisode. Le format est toujours json. Ex: recordings/train_%.json")
def main(unattended, episodes, agent, windowsize, renderfps, record):
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

    if not record is None:
        configs.graphics.caption += " - recording"

    application = TrainApplication(configs)

    if not record is None:
        application.agent = AgentActionRecorder(application.agent, record)

    application.run()


if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    main()
