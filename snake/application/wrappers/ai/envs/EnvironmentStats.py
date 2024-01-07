import gymnasium as gym
import numpy as np
import pandas as pd

from core import Delegate
from tqdm import tqdm


_ID = "Id"
_EPISODE = "Episode"
_EPISODE_LENGTH = "EpisodeLength"
_SCORE = "Score"
_CUM_REWARD = "CumulativeReward"

class EnvironmentStats(gym.ObservationWrapper):
    """
    Encapsule un environment pour afficher ses statistiques
    """
    def __init__(self, env, tqdmBasePosition, id, filename, showStats=True):
        gym.ObservationWrapper.__init__(self, env)

        self._episode = -1
        self._id = id
        self._filename = filename.replace("%", "stats")
        self._stats = None
        self._maxStats = None
        self._maxEpisode = None
        self._saved = False
        self._showStats = showStats
        self._newMaxStatsDelegate = Delegate()

        if self._showStats:
            self._episodeProgress = tqdm(bar_format="Max episode length: {desc} at {unit}",
                                        position=tqdmBasePosition)

            self._rewardProgress = tqdm(bar_format="Max cum. reward: {desc} at {unit}",
                                        position=tqdmBasePosition + 1)

            self._scoreProgress = tqdm(bar_format="Max score: {desc} at {unit}",
                                       position=tqdmBasePosition + 2)

        self.observation_space = env.observation_space

    @property
    def statsDataFrame(self):
        return self._stats

    @property
    def newMaxStatsDelegate(self):
        return self._newMaxStatsDelegate

    def reset(self, *args, seed=None, options=None):
        if not options is None and "episode" in options:
            self._episode = options["episode"]
        else:
            self._episode += 1

        self.save()
        self._newEpisode()

        return self.env.reset(*args, seed=seed, options=options)

    def step(self, *args):
        observations, reward, terinated, truncated, info = self.env.step(*args)

        self._stats.loc[0, _EPISODE_LENGTH] += 1
        self._stats.loc[0, _SCORE] = observations["score"]
        self._stats.loc[0, _CUM_REWARD] += reward

        forceUpdate = False
        if self._maxStats is None:
            self._maxStats = self._stats.copy()
            self._maxEpisode = self._newDataFrame()
            self._maxEpisode.iloc[:,:] = self._episode
            forceUpdate = True

        greater = self._stats > self._maxStats
        greaterEqual = self._stats >= self._maxStats

        self._maxStats[greaterEqual] = self._stats[greaterEqual]
        self._maxEpisode[greaterEqual] = self._episode

        if greaterEqual.iloc[0,2:4].any(axis=0) or forceUpdate:
            self._update()

        if greater.loc[0, _SCORE]:
            self._newMaxStatsDelegate()

        return observations, reward, terinated, truncated, info

    def observation(self, observation):
        return observation

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        self.save()

    def save(self):
        if not self._stats is None and \
           not self._filename is None:
            if self._saved:
                self._stats.to_csv(self._filename, mode="a", index=False, header=False)
            else:
                self._stats.to_csv(self._filename, mode="w", index=False)

            self._saved = True

    def _update(self):
        if self._showStats:
            self._episodeProgress.set_description_str(str(self._maxStats.loc[0, _EPISODE_LENGTH]))
            self._episodeProgress.unit = str(self._maxEpisode.loc[0, _EPISODE_LENGTH])

            self._rewardProgress.set_description_str(str(self._maxStats.loc[0, _CUM_REWARD]))
            self._rewardProgress.unit = str(int(self._maxEpisode.loc[0, _CUM_REWARD]))

            self._scoreProgress.set_description_str(str(self._maxStats.loc[0, _SCORE]))
            self._scoreProgress.unit = str(self._maxEpisode.loc[0, _SCORE])

    def _newEpisode(self):
        self._stats = self._newDataFrame()

    def _newDataFrame(self):
        return pd.DataFrame([[self._id, self._episode, 0, 0, 0.0]],
                            columns=[_ID, _EPISODE, _EPISODE_LENGTH, _SCORE, _CUM_REWARD])
