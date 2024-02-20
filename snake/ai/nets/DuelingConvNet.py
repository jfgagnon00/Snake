from torch import concatenate
from torch.nn import Linear, \
                    Module, \
                    Sequential, \
                    Conv2d, \
                    Flatten, \
                    LeakyReLU, \
                    MaxPool2d

class _DuelingConvNet(Module):
    def __init__(self, width, height, numChannels, numInputs, numOutputs):
        super().__init__()

        self._conv = Sequential(
            Conv2d(numChannels, 16, 5, padding="same"),
            LeakyReLU(),

            MaxPool2d(2, stride=2),

            Conv2d(16, 32, 3, padding="same"),
            LeakyReLU(),

            Flatten()
        )

        w2 = width // 2
        h2 = height // 2

        self._value = Sequential(
            Linear(w2 * h2 * 32  + numInputs, 128),
            LeakyReLU(),

            Linear(128, 128),
            LeakyReLU(),

            Linear(128, 1),
        )

        self._advantage = Sequential(
            Linear(w2 * h2 * 32 + numInputs, 128),
            LeakyReLU(),

            Linear(128, 128),
            LeakyReLU(),

            Linear(128, numOutputs),
        )

    def forward(self, x0, x1):
        features = self._conv(x0)

        if not x1 is None:
            features = concatenate((features, x1), dim=1)

        value = self._value(features)
        advantage = self._advantage(features)

        return value + (advantage - advantage.mean())
