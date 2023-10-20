from tqdm import tqdm


class EnvironmentStats():
    """
    Encapsule un environment pour afficher ses statistiques
    """
    def __init__(self, env, position):
        self._env = env
        self._episode = -1

        self._maxCumulativeReward = -1000
        self._maxCumulativeRewardEpisode = -1

        self._maxEpisodeLength = -1
        self._maxEpisodeLengthEpisode = -1

        self._maxLength = -1
        self._maxLengthEpisode = -1

        self._episodeProgress = tqdm(bar_format="Max episode length: {desc} at {unit}",
                                     position=position)
        self._rewardProgress = tqdm(bar_format="Max cum. reward: {desc} at {unit}",
                                    position=position + 1)
        self._lengthProgress = tqdm(bar_format="Max length: {desc} at {unit}",
                                    position=position + 2)
        self._resetStats()

    def reset(self, **kwargs):
        self._episode += 1
        self._resetStats()
        return self._env.reset(**kwargs)

    def step(self, *args):
        observation, reward, terinated, truncated, info = self._env.step(*args)

        self._episodeLength += 1
        if self._episodeLength > self._maxEpisodeLength:
            self._maxEpisodeLength = self._episodeLength
            self._maxEpisodeLengthEpisode = self._episode

        self._cumulativeReward += reward
        if self._cumulativeReward > self._maxCumulativeReward:
            self._maxCumulativeReward = self._cumulativeReward
            self._maxCumulativeRewardEpisode = self._episode

        if observation["length"] > self._maxLength:
            self._maxLength = observation["length"]
            self._maxLengthEpisode = self._episode


        return observation, reward, terinated, truncated, info

    def render(self):
        self._update()
        self._env.render()

    def close(self):
        self._env.close()

    def _update(self):
        self._episodeProgress.set_description_str(str(self._maxEpisodeLength))
        self._episodeProgress.unit = str(self._maxEpisodeLengthEpisode)

        self._rewardProgress.set_description_str(str(self._maxCumulativeReward))
        self._rewardProgress.unit = str(self._maxCumulativeRewardEpisode)

        self._lengthProgress.set_description_str(str(self._maxLength))
        self._lengthProgress.unit = str(self._maxLengthEpisode)

    def _resetStats(self):
        self._cumulativeReward = 0
        self._episodeLength = 0
