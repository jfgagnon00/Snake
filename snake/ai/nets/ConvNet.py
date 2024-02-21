from torch import concatenate
from torch.nn import Linear, \
                    Module, \
                    Sequential, \
                    Conv2d, \
                    Flatten, \
                    LeakyReLU, \
                    MaxPool2d
from torch.nn.functional import leaky_relu


class _ConvNet(Module):
    def __init__(self, width, height, numChannels, numInputs, numOutputs):
        super().__init__()

        self._convs = Sequential(
            Conv2d(numChannels, 16, 5, padding="same"),
            LeakyReLU(),

            MaxPool2d(2, stride=2),

            Conv2d(16, 32, 3, padding="same"),
            LeakyReLU(),

            Flatten()
        )

        w2 = width // 2
        h2 = height // 2

        size = 0
        size += w2 * h2 * 32
        size += numInputs

        self._linear = Sequential(
            Linear(size, 128),
            LeakyReLU(),

            Linear(128, 128),
            LeakyReLU(),

            Linear(128, numOutputs),
        )

    def forward(self, x0, x1):
        features = self._convs(x0)
        if not x1 is None:
            features = concatenate((features, x1), dim=1)
        return self._linear(features)
