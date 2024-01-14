from collections import deque
from numpy import array
from numpy.random import choice

from .ReplayBuffer import _ReplayBuffer


class _PriorityReplayBuffer(_ReplayBuffer):
    def __init__(self, maxlen, alpha, beta, betaAnnealingSteps):
        super().__init__(maxlen)

        self._alpha = alpha
        self._beta = beta
        self._betaAnnealingSteps = betaAnnealingSteps
        self._maxPriority = 1
        self._priorities = deque(maxlen=maxlen)

    def append(self, state, action, newState, reward, done):
        super().append(state, action, newState, reward, done)
        self._priorities.append(self._maxPriority)

    def sample(self, count):
        # TODO: implementation tres lente, a refaire
        props = array(self._priorities) ** self._alpha
        props = props / props.sum()

        size = len(self)
        indices = choice(size, count, p=props)

        beta = self._beta + (1.0 - self._beta) * size / self._betaAnnealingSteps
        beta = min(1.0, beta)
        weights = (size * props[indices]) ** -beta
        weights = weights / weights.max()

        return *self._getitems(indices), weights, indices

    def updatePriorities(self, indices, errors):
        for i, j in enumerate(indices):
            error = errors[i, 0]
            self._maxPriority = max(error, self._maxPriority)
            self._priorities[j] = error
