import ai # importe l'environnement "snake/SnakeEnvironment-v0"

from application.wrappers.ai.envs import EnvironmentStats
from gymnasium import make as gym_Make
from gymnasium.wrappers import TimeLimit as gym_TimeLimit
from tqdm import trange


class ApplicationTrain():
    """
    Responsable de l'entrainement des agents
    """
    def __init__(self, configs, agent):
        self._episodes = configs.train.episodes
        self._agent = agent
        self._env = gym_Make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics)
        self._envStats = EnvironmentStats(self._env, 1)
        self._env = self._envStats

        if configs.train.episodeMaxLen > 0:
            self._env = gym_TimeLimit(self._env,
                                      max_episode_steps=configs.train.episodeMaxLen)

    @property
    def envStats(self):
        return self._envStats

    def run(self):
        episodesIt = trange(self._episodes, desc="Episodes", position=0)
        for e in episodesIt:
            done = False
            self._agent.reset()
            observations, _ = self._env.reset()

            while not done:
                action = self._agent.getAction(observations)

                newObservations, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                self._env.render()
                self._agent.train(observations, action, newObservations, reward, done)

                observations = newObservations

            self._agent.onEpisodeDone(e)

        self._env.close()
