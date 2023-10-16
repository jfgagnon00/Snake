"""
Responsable de l'entrainement des agents
"""


import ai # importe l'environnement "snake/SnakeEnvironment-v0"
import ai.agents as agents
import click
import os

from configs import configsCreate
from tqdm import tqdm
from wrappers.ai.agents import AgentActionRecorder
from gymnasium import make as gym_Make
from gymnasium.wrappers import TimeLimit as gym_TimeLimit


class TrainApplication():
    def __init__(self, configs):
        self._episodes = configs.train.episodes
        self.agent = self._createAgent(configs.train, configs.simulation)
        self._env = gym_Make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)

        if configs.train.episodeMaxLen > 0:
            self._env = gym_TimeLimit(self._env,
                                      max_episode_steps=configs.train.episodeMaxLen)

    def run(self):
        for e in tqdm(range(self._episodes)):
            done = False
            self.agent.reset()
            state, _ = self._env.reset()

            while not done:
                action = self.agent.getAction(state)

                newState, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                self._env.render()
                self.agent.train(state, action, newState, reward, done)

                state = newState

            last = e == (self._episodes - 1)
            self.agent.onSimulationDone(last)

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
              help="Train sans rendu.")
@click.option("--episodes",
              "-e",
              type=int,
              help="Nombre d'épisodes pour l'entrainement.")
@click.option("--episodeMaxLen",
              "-eml",
              type=int,
              help="Longueur maximale pour un épisode.")
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
@click.option("--recordN",
              "-rn",
              type=int,
              help="Si record est spécifié, enregistre un épisode tout les N épisodes.")
def main(unattended,
         episodes,
         episodemaxlen,
         agent,
         windowsize,
         renderfps,
         record,
         recordn):
    configs = configsCreate("config_overrides.json")
    configs.train.unattended = unattended

    if not episodes is None and episodes > 0:
        configs.train.episodes = episodes

    if not episodemaxlen is None and episodemaxlen > 0:
        configs.train.episodeMaxLen = episodemaxlen

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
        application.agent = AgentActionRecorder(application.agent, record, recordn)

    application.run()


if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    main()
