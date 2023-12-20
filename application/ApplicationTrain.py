import ai # importe l'environnement "snake/SnakeEnvironment-v0"

from application.wrappers.ai.envs import EnvironmentStats
from datetime import datetime
from gymnasium import make as gym_Make
from gymnasium.wrappers import TimeLimit as gym_TimeLimit
from tqdm import trange


class ApplicationTrain():
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
                            graphicsConfig=configs.graphics)

        if configs.train.episodeMaxLen > 0:
            self._timeLimit = gym_TimeLimit(self._env,
                                      max_episode_steps=configs.train.episodeMaxLen)
            self._env = self._timeLimit
        else:
            self._timeLimit = None

        self._envStats = EnvironmentStats(self._env, 1, 0, statsFilename, showStats=True)
        self._env = self._envStats

    @property
    def envStats(self):
        return self._envStats

    def run(self):
        start = None
        episodesIt = trange(self._episodes, desc="Episodes", position=0)
        for e in episodesIt:
            done = False
            self._agent.reset()
            observations, _ = self._env.reset(options={"episode":e})

            self._agent.onEpisodeBegin(e, self._envStats.statsDataFrame)

            if start is None:
                start = datetime.now()

            dt = datetime.now() - start
            self._envStats.statsDataFrame.loc[0, "ElapseSeconds"] = dt.seconds

            while not done:
                action = self._agent.getAction(observations)

                newObservations, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                if not self._timeLimit is None and reward > 0:
                    # time limit quand il n'y a pas progression dans le jeu
                    # reset quand il y en a
                    self._timeLimit._elapsed_steps = 0

                self._env.render()
                self._agent.train(observations, action, newObservations, reward, done)

                observations = newObservations

            self._agent.onEpisodeDone(e)

        self._env.close()
