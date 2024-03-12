import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from snake.core import Delegate


_ID = "Id"
_EPISODE = "Episode"
_EPISODE_LENGTH = "EpisodeLength"
_SCORE = "Score"
_CUM_REWARD = "CumulativeReward"
_CAUSE_OF_TERMINATION = "CauseOfTermination"

_ROLLING_MEAN = 100
_SAMPLES = -500

class EnvironmentStats(gym.ObservationWrapper):
    """
    Encapsule un environment pour afficher ses statistiques
    """
    def __init__(self, env, tqdmBasePosition, id, filename, showStats=True):
        gym.ObservationWrapper.__init__(self, env)

        self._episode = -1
        self._id = id
        self._filename = filename.replace("%", "stats") if filename else None
        self._currentEpisodeStats = None
        self._maxStats = None
        self._maxEpisode = None
        self._saved = False
        self._showStats = showStats
        self._newMaxStatsDelegate = Delegate()

        if self._showStats:
            self._lastUpate = datetime.now()
            self._allEpisodeStats = None
            self._constructPlot()
            plt.show(block=False)

        self.observation_space = env.observation_space

        env.unwrapped.trappedDelegate.register(lambda: self._onTermination("Trapped"))
        env.unwrapped.winDelegate.register(lambda: self._onTermination("Win"))

    @property
    def statsDataFrame(self):
        return self._currentEpisodeStats

    @property
    def newMaxStatsDelegate(self):
        return self._newMaxStatsDelegate

    def reset(self, *args, seed=None, options=None):
        if not options is None and "episode" in options:
            self._episode = options["episode"]
        else:
            self._episode += 1

        self.save()

        if self._showStats and not self._currentEpisodeStats is None:
            if self._allEpisodeStats is None:
                self._allEpisodeStats = self._currentEpisodeStats
            else:
                self._allEpisodeStats = pd.concat([self._allEpisodeStats, self._currentEpisodeStats], axis=0)

        self._newEpisode()

        if self._showStats:
            if self._allEpisodeStats is None:
                self._updatePlot(self._currentEpisodeStats)
            else:
                self._updatePlot(self._allEpisodeStats)

        return self.env.reset(*args, seed=seed, options=options)

    def step(self, *args):
        observations, reward, terinated, truncated, infos = self.env.step(*args)

        if truncated:
            self._onTermination("EpisodeTruncated")

        self._currentEpisodeStats.loc[0, _EPISODE_LENGTH] += 1
        self._currentEpisodeStats.loc[0, _SCORE] = infos["score"]
        self._currentEpisodeStats.loc[0, _CUM_REWARD] += reward

        forceUpdate = False
        if self._maxStats is None:
            self._maxStats = self._currentEpisodeStats.copy()
            self._maxEpisode = self._newDataFrame()
            self._maxEpisode.iloc[:,:] = self._episode
            forceUpdate = True

        greater = self._currentEpisodeStats > self._maxStats
        greaterEqual = self._currentEpisodeStats >= self._maxStats

        self._maxStats[greaterEqual] = self._currentEpisodeStats[greaterEqual]
        self._maxEpisode[greaterEqual] = self._episode

        if greater.loc[0, _CUM_REWARD]:
            self._newMaxStatsDelegate()

        return observations, reward, terinated, truncated, infos

    def observation(self, observation):
        return observation

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.save()

    def save(self):
        if not self._currentEpisodeStats is None and \
           not self._filename is None:
            if self._saved:
                self._currentEpisodeStats.to_csv(self._filename, mode="a", index=False, header=False)
            else:
                self._currentEpisodeStats.to_csv(self._filename, mode="w", index=False)

            self._saved = True

    def _updatePlot(self, df):
        t = datetime.now()
        dt = t - self._lastUpate

        if dt.total_seconds() > 6:
            self._lastUpate = t

            episode = df.Episode
            cot = df.CauseOfTermination
            score = df.Score
            trainError = df.TrainLossMean

            EnvironmentStats._updateScatter(self._score, episode, score, "Score")
            EnvironmentStats._updatePiePlot(self._causeOfTemination, cot, "Cause Of Termination")
            EnvironmentStats._updateScatter(self._trainError, episode, trainError, "Train Error (Mean)")

            self._figure.canvas.draw()
            self._figure.canvas.flush_events()

            plt.tight_layout()

    @staticmethod
    def _updateScatter(ax, x, y, title, size=5):
        xx = x[_SAMPLES:]
        yy = y[_SAMPLES:]
        yym = y.rolling(_ROLLING_MEAN).mean()[_SAMPLES:]
        ax.cla()
        ax.plot(xx, yym, color="blue")
        ax.scatter(xx, yy, s=size, color="orange")
        ax.set_title(title)
        ax.grid()

    @staticmethod
    def _updatePiePlot(ax, y, title):
        samples = y[_SAMPLES:].value_counts()

        ax.cla()
        ax.pie(samples.values,
               labels=samples.index,
               autopct="%2.1f%%",
               explode=[0.05] * len(samples))
        ax.set_title(title)

    @staticmethod
    def _updateBarPlot(ax, dict_, title):
        size = len(dict_)
        ax.cla()
        ax.bar(range(size), dict_.values())
        ax.set_title(f"{title} - {size}")

    def _constructPlot(self):
        layout = [
            ["A", "B"],
            ["Z", "Z"],
        ]

        matplotlib.rcParams['toolbar'] = 'None'

        self._figure, ax = plt.subplot_mosaic(layout, figsize=(7, 6), height_ratios=[2, 3])
        self._figure.canvas.manager.set_window_title(f"Stats - dernier {_ROLLING_MEAN} samples")

        self._score = ax["A"]
        self._causeOfTemination = ax["B"]
        self._trainError = ax["Z"]

        plt.tight_layout()

    def _newEpisode(self):
        self._currentEpisodeStats = self._newDataFrame()

    def _newDataFrame(self):
        return pd.DataFrame([[self._id, self._episode, 0, 0, 0.0, ""]],
                            columns=[_ID, _EPISODE, _EPISODE_LENGTH, _SCORE, _CUM_REWARD, _CAUSE_OF_TERMINATION])

    def _onTermination(self, cause):
        self._currentEpisodeStats.loc[0, _CAUSE_OF_TERMINATION] = cause