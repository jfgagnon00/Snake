import ai # importe l'environnement "snake/SnakeEnvironment-v0"
import ai.agents as agents

from application.wrappers.ai.envs import EnvironmentStats
from gymnasium import make as gym_Make
from gymnasium.wrappers import TimeLimit as gym_TimeLimit
from tqdm import trange


class ApplicationTrain():
    """
    Responsable de l'entrainement des agents
    """
    def __init__(self, configs):
        self._episodes = configs.train.episodes
        self.agent = self._createAgent(configs.train, configs.simulation)
        self._env = gym_Make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)

        self._env = EnvironmentStats(self._env, 1)

        if configs.train.episodeMaxLen > 0:
            self._env = gym_TimeLimit(self._env,
                                      max_episode_steps=configs.train.episodeMaxLen)

    def run(self):
        episodesIt = trange(self._episodes,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                            desc="Episodes")
        for e in episodesIt:
            done = False
            self.agent.reset()
            observation, _ = self._env.reset()

            while not done:
                action = self.agent.getAction(observation)

                newObservation, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                self._env.render()
                self.agent.train(observation, action, newObservation, reward, done)

                observation = newObservation

            last = e == (self._episodes - 1)
            self.agent.onSimulationDone(last)

        self._env.close()

    def _createAgent(self, trainConfig, simulatinConfig):
        # instantier un agent a partir d'un string
        # limiter a ai.agents pour le moment
        agent_class = getattr(agents, trainConfig.agent)
        return agent_class(trainConfig, simulatinConfig)
