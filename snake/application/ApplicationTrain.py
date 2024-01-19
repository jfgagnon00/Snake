import snake.ai # importe l'environnement "snake/SnakeEnvironment-v0"

from datetime import datetime
from gymnasium import make as gym_Make
from tqdm import trange
from snake.application.wrappers.ai.envs import EnvironmentStats, \
                                               EnvironmentStackedOccupancyGrid


class ApplicationTrain(object):
    """
    Responsable de l'entrainement des agents
    """
    def __init__(self, configs, agent, statsFilename=None):
        self._episodes = configs.train.episodes
        self._agent = agent
        self._env = gym_Make("snake/SnakeEnvironment-v0",
                            renderMode = None if configs.train.unattended else "human",
                            environmentConfig=configs.environment,
                            simulationConfig=configs.simulation,
                            graphicsConfig=configs.graphics,
                            trainConfig=configs.train)
        self._envStats = EnvironmentStats(self._env,
                                          1,
                                          0,
                                          statsFilename,
                                          showStats=configs.train.showStats)
        self._envStats.newMaxStatsDelegate.register(self._onNewMaxStats)
        self._env = self._envStats

        if configs.train.useFrameStack:
            self._env = EnvironmentStackedOccupancyGrid(self._env,
                                                        configs.train.frameStack)

    @property
    def envStats(self):
        return self._envStats

    def run(self):
        self._forceSaveEpisode = False
        episodesIt = trange(self._episodes, desc="Episodes", position=0)
        for e in episodesIt:
            done = False
            self._agent.reset()
            observations, _ = self._env.reset(options={"episode":e})

            self._agent.onEpisodeBegin(e, self._envStats.statsDataFrame)
            start = datetime.now()

            while not done:
                action = self._agent.getAction(observations)

                newObservations, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                self._env.render()
                self._agent.train(observations, action, newObservations, reward, done)

                observations = newObservations

            dt = datetime.now() - start
            self._envStats.statsDataFrame.loc[0, "EpisodeDuration"] = dt.total_seconds()
            self._agent.onEpisodeDone(e)
            self._onEpisodeDone()

        self._env.close()

    def _onNewMaxStats(self):
        self._forceSaveEpisode = True

    def _onEpisodeDone(self):
        if self._forceSaveEpisode:
            self._forceSaveEpisode = False
            self._agent.save()
