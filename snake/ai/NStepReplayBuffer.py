from collections import deque

from .ReplayBuffer import _ReplayBuffer


class _NStepReplayBuffer(_ReplayBuffer):
    def __init__(self, maxlen, gamma, nStep):
        super().__init__(maxlen)
        self._gamma = gamma
        self._nStep = nStep
        self._nStepBuffer = deque(maxlen=nStep)

    @property
    def nStep(self):
        return self._nStep

    def append(self, state, action, newState, reward, done):
        self._nStepBuffer.append((state, action, newState, reward, done))

        if len(self._nStepBuffer) == self._nStep:
            G = 0

            for state_, action_, newState_, reward_, done_ in reversed(self._nStepBuffer):
                G = reward_ + self._gamma * G

            super().append(state_, action_, newState_, G, done_)
            self._nStepBuffer.popleft()

        if done and len(self._nStepBuffer) > 0:
            G = 0

            for state_, action_, newState_, reward_, done_ in reversed(self._nStepBuffer):
                G = reward_ + self._gamma * G
                super().append(state_, action_, newState_, G, done_)

            self._nStepBuffer.clear()
