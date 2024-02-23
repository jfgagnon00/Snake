from .ReplayBuffer import _ReplayBuffer


class _HindsightReplayBuffer(_ReplayBuffer):
    def __init__(self, maxlen):
        super().__init__(maxlen)

    def append(self, state, action, newState, reward, done, hindsightInfo):
        self._episode.append(hindsightInfo)
        super().append(state, action, newState, reward, done)

    def onEpisodeBegin(self):
        self._episode = []

    def onEpisodeDone(self):
        pass
