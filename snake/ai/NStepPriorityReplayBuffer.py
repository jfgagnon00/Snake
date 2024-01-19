from collections import deque

from .PriorityReplayBuffer import _PriorityReplayBuffer


class _NStepPriorityReplayBuffer(_PriorityReplayBuffer):
    def __init__(self, maxlen, alpha, beta, betaAnnealingSteps, gamma, nStep):
        super().__init__(maxlen, alpha, beta, betaAnnealingSteps)
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
