from torch import concatenate
from torch.nn import Linear, \
                    Module, \
                    Sequential, \
                    LeakyReLU


class _LinearNet(Module):
    def __init__(self, numInputs, hiddenLayers, numOutput):
        super().__init__()

        self._net = Sequential()

        prevSize = numInputs
        for size in hiddenLayers:
            self._net.append(Linear(prevSize, size))
            self._net.append(LeakyReLU())
            prevSize = size

        self._net.append(Linear(prevSize, numOutput))

    def forward(self, x0, x1):
        if x1 is None:
            out = x0
        else:
            out = concatenate((x0, x1), dim=1)
        return self._net(out)
