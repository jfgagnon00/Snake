from game import GameAction
from random import random, choice
from torch import argmax, from_numpy, Tensor
from torch.nn import L1Loss, Linear, Module, ReLU, Sequential, MSELoss
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

        self._gamma = trainConfig.gamma
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay

        self._model = _QNet(numInputs,
                          [x for x in div2Generator(numInputs, numOutput * 2)],
                          numOutput)
        self._optimizer = Adam(self._model.parameters(), lr=trainConfig.lr)
        self._lossFnc = MSELoss()
        self._gameActions = list(GameAction)

        self._iteration = 0

    def train(self, state, action, newState, reward, done):
        x = self._stateToTensor(state)
        q = self._model(x)
        q_target = q.clone()
        if done:
            q_target[action] = reward
        else:
            x_new = self._stateToTensor(newState)
            q_new = self._model(x_new)
            q_target[action] = reward + self._gamma * q_new[action]

        self._optimizer.zero_grad()
        loss = self._lossFnc(q_target, q)
        loss.backward()
        self._optimizer.step()

    def getAction(self, state):
        if random() < self._epsilon:
            gameAction = choice(self._gameActions)
        else:
            x = self._stateToTensor(state)
            actions = self._model(x)
            gameAction = argmax(actions).item()
            gameAction = self._gameActions[gameAction]

        return GameAction(gameAction)

    def onSimulationDone(self, *args):
        self._iteration += 1
        eps = self._epsilon * self._epsilonDecay
        eps = max(eps, 0.01)
        # print(self._iteration, ":", self._epsilon, eps, self._epsilonDecay,)
        self._epsilon = eps

    def _stateToTensor(self, state):
        x = state["occupancy_grid"]
        x = from_numpy(x.reshape(-1))
        x = convert_image_dtype(x)
        return x
