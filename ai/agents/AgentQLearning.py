from game import GameAction
from random import random
from torch import argmax, from_numpy
from torch.nn import L1Loss, Linear, Module, ReLU, Sequential
from torch.optim import Adam
from torchvision.transforms.functional import convert_image_dtype

from .AgentBase import AgentBase


class _QNet(Module):
    def __init__(self, numInputs, hiddenLayers, numOutput):
        super().__init__()

        self._net = Sequential()

        prevSize = numInputs
        for size in hiddenLayers:
            self._net.append(Linear(prevSize, size))
            self._net.append(ReLU())
            prevSize = size

        self._net.append(Linear(prevSize, numOutput))

    def forward(self, x):
        return self._net(x)

class _QTrainer():
    def __init__(self, model, trainConfig):
        self._optimizer = Adam(model.parameters(), lr=trainConfig.lr)
        self._loss = L1Loss()

    def train(self, state, newState, reward, done):
        pass

class Agent47(AgentBase):
    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        numInputs = simulationConfig.gridWidth * simulationConfig.gridHeight
        numOutput = len(GameAction)

        def div2Generator(x, stop):
            while True:
                x //= 2
                if x > stop:
                    yield x
                else:
                    break

        self._model = _QNet(numInputs,
                          [x for x in div2Generator(numInputs, numOutput * 2)],
                          numOutput)
        self._trainer = _QTrainer(self._model, trainConfig)

    def train(self, state, newState, reward, done):
        self._trainer.train(state, newState, reward, done)

    def getAction(self, state):
        x =  state["occupancy_grid"]
        x = from_numpy(x.reshape(-1))
        x = convert_image_dtype(x)

        actions = self._model(x)
        gameAction = argmax(actions).item()

        return GameAction.fromInt(gameAction)
