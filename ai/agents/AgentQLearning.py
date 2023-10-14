from game import GameAction
from torch.nn import Linear, Module, ReLU, Sequential

from .AgentBase import AgentBase


class QNet(Module):
    def __init__(self, numInputs, numOutput, numHidden, hiddenSize):
        super().__init__()

        self._net = Sequential()

        self._net.append(Linear(numInputs, hiddenSize))
        self._net.append(ReLU())

        for _ in range(numHidden):
            self._net.append(Linear(hiddenSize, hiddenSize))
            self._net.append(ReLU())

        self._net.append(Linear(hiddenSize, numOutput))

    def forward(self, x):
        return self._net.apply(x)


def QTrainer():
    pass


class AgentQLearning(AgentBase):
    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__(trainConfig, simulationConfig)
        self._qnet = QNet(10, len(GameAction), 1)

    def reset(self):
        pass

    def getAction(self, *args):
        pass

    def onSimulationDone(self):
        pass