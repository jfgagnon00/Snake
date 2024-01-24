from torch.nn import Linear, \
                    Module, \
                    Sequential, \
                    Conv2d, \
                    Flatten, \
                    LeakyReLU

class _DuelingConvNet(Module):
    def __init__(self, width, height, numInputs, numOutputs):
        super().__init__()

        self._conv = Sequential(
            Conv2d(numInputs, 16, 3, padding="same"),
            LeakyReLU(),
            Flatten()
        )

        self._value = Sequential(
            Linear(width * height * 16, 128),
            LeakyReLU(),
            Linear(128, 1),
        )

        self._advantage = Sequential(
            Linear(width * height * 16, 128),
            LeakyReLU(),
            Linear(128, numOutputs),
        )

    def forward(self, x):
        features = self._conv(x)
        value = self._value(features)
        advantage = self._advantage(features)
        return value + (advantage - advantage.mean())
