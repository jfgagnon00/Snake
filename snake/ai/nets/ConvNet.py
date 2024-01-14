from torch.nn import Linear, \
                    Module, \
                    Sequential, \
                    Conv2d, \
                    Flatten, \
                    LeakyReLU


class _ConvNet(Module):
    def __init__(self, width, height, numInputs, numOutputs):
        super().__init__()

        self._net = Sequential()

        self._net.append(Conv2d(numInputs, 16, 3, padding=1))
        self._net.append(LeakyReLU())

        self._net.append(Conv2d(16, 32, 3, padding=1))
        self._net.append(LeakyReLU())

        self._net.append(Flatten())

        self._net.append(Linear(32 * width * height, numOutputs))

    def forward(self, x):
        return self._net(x)

