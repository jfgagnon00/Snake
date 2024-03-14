from torch import concatenate
from torch.nn import Linear, \
                    Module, \
                    Sequential, \
                    Conv2d, \
                    Flatten, \
                    ReLU, \
                    LeakyReLU, \
                    MaxPool2d


class _AlphaGoZeroConvNet(Module):
    def __init__(self, width, height, numChannels, numInputs, numOutputs):
        super().__init__()

        self._convs = Sequential(
            Conv2d(numChannels, 16, 3, padding="same"),
            LeakyReLU(),

            Conv2d(16, 16, 3, padding="same"),
            LeakyReLU(),

            MaxPool2d(2, stride=2),

            Flatten()
        )

        w2 = width // 2
        h2 = height // 2

        size = 0
        size += w2 * h2 * 16
        size += numInputs

        self._p = Sequential(
            Linear(size, 200),
            LeakyReLU(),

            Linear(200, 100),
            ReLU(),

            Linear(100, numOutputs),
        )

        self._v = Sequential(
            Linear(size, 200),
            LeakyReLU(),

            Linear(200, 100),
            LeakyReLU(),

            Linear(100, 1),
        )

    def forward(self, x0, x1):
        features = self._convs(x0)

        if not x1 is None:
            features = concatenate((features, x1), dim=1)

        logits_ = self._p(features)
        v = self._v(features)

        return logits_, v
