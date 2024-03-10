from collections import deque
from numpy.random import choice


class _ReplayBuffer(object):
    def __init__(self, maxlen):
        self._buffer = deque(maxlen=maxlen)

    def clear(self):
        self._buffer.clear()

    def append(self, sample):
        self._buffer.append(sample)

    def sample(self, count):
        indices = choice(len(self), count)
        return self.getitems(indices)

    def getitems(self, indices):
        elements = [self._buffer[i] for i in indices]
        return zip(*elements)

    def __len__(self):
        return len(self._buffer)
