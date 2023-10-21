import numpy as np
import pandas as pd

from tqdm import tqdm


_ID = "Id"
_EPISODE = "Episode"
_EPISODE_LENGTH = "EpisodeLength"
_SNAKE_LENGTH = "SnakeLength"
_CUM_REWARD = "CumulativeReward"

class EnvironmentStats():
    """
    Encapsule un environment pour afficher ses statistiques
    """
    def __init__(self, env, tqdmBasePosition, id, filename):
        self._env = env
        self._episode = -1
        self._id = id
        self._filename = filename
        self._stats = None
        self._maxStats = None
        self._maxEpisode = None
        self._saved = False

        self._episodeProgress = tqdm(bar_format="Max episode length: {desc} at {unit}",
                                     position=tqdmBasePosition)

        self._rewardProgress = tqdm(bar_format="Max cum. reward: {desc} at {unit}",
                                    position=tqdmBasePosition + 1)

        self._lengthProgress = tqdm(bar_format="Max length: {desc} at {unit}",
                                    position=tqdmBasePosition + 2)

    def reset(self, options=None):
        if not options is None and "episode" in options:
            self._episode = options["episode"]
        else:
            self._episode += 1

        self.save()
        self._newEpisode()

        return self._env.reset(options=options)

    def step(self, *args):
        observations, reward, terinated, truncated, info = self._env.step(*args)

        # self._stats.loc[0, _EPISODE_LENGTH] += 1
        # self._stats.loc[0, _SNAKE_LENGTH] = observations["length"]
        # self._stats.loc[0, _CUM_REWARD] += reward

        # forceUpdate = False
        # if self._maxStats is None:
        #     self._maxStats = self._stats.copy()
        #     self._maxEpisode = self._newDataFrame()
        #     self._maxEpisode.loc[:,:] = self._episode
        #     forceUpdate = True

        # greater = self._stats > self._maxStats

        # self._maxStats[greater] = self._stats[greater]
        # self._maxEpisode[greater] = self._episode

        # if greater.iloc[0,2:].any(axis=0) or forceUpdate:
        #     # self._update()
        #     pass

        return observations, reward, terinated, truncated, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
        self.save()

    def save(self):
        return
        if not self._stats is None and \
           not self._filename is None:
            if self._saved:
                self._stats.to_csv(self._filename, mode="a", index=False, header=False)
            else:
                self._stats.to_csv(self._filename, mode="w", index=False)

            self._saved = True

    def _update(self):
        self._episodeProgress.set_description_str(str(self._maxStats.loc[0, _EPISODE_LENGTH]))
        self._episodeProgress.unit = str(self._maxEpisode.loc[0, _EPISODE_LENGTH])

        self._rewardProgress.set_description_str(str(self._maxStats.loc[0, _CUM_REWARD]))
        self._rewardProgress.unit = str(self._maxEpisode.loc[0, _CUM_REWARD])

        self._lengthProgress.set_description_str(str(self._maxStats.loc[0, _SNAKE_LENGTH]))
        self._lengthProgress.unit = str(self._maxEpisode.loc[0, _SNAKE_LENGTH])

    def _newEpisode(self):
        self._stats = self._newDataFrame()

    def _newDataFrame(self):
        return pd.DataFrame([[self._id, self._episode, 0, 0, 0.0]],
                            columns=[_ID, _EPISODE, _EPISODE_LENGTH, _SNAKE_LENGTH, _CUM_REWARD])
