import numpy as np

from game import GameAction
from numpy.linalg import norm
from numpy.random import choice as np_choice
from random import random, choice as py_choice
from torch import argmax, from_numpy
from torch.nn import L1Loss, Linear, Module, MSELoss, ReLU, Sequential
from torch.nn.functional import softmax
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

        numInputs = 6 # simulationConfig.gridWidth * simulationConfig.gridHeight
        numOutput = len(GameAction)

        def div2Generator(x, stop):
            while True:
                x //= 2
                if x > stop:
                    yield x
                else:
                    break

        self._alpha = trainConfig.alpha
        self._gamma = trainConfig.gamma
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay
        self._episodes = trainConfig.episodes
        self._episode = 0

        self._model = _QNet(numInputs,
                        #   [x for x in div2Generator(numInputs, numOutput * 2)],
                          [numInputs for _ in range(8)],
                          numOutput)
        self._optimizer = Adam(self._model.parameters(), lr=trainConfig.lr)
        self._lossFnc = MSELoss()
        self._gameActions = list(GameAction)

    def getAction(self, state):
        if random() < self._epsilon:
            gameAction = np_choice(self._gameActions)
        else:
            x = self._stateToTensor(state)
            actions = self._model(x)
            intAction = actions.argmax().item()
            gameAction = self._gameActions[intAction]
            # p = softmax(actions, dim=0).detach().numpy()
            # gameAction = np_choice(self._gameActions, p=p)

        # gameAction = np_choice(self._gameActions)

        return GameAction(gameAction)

    def onEpisodeDone(self, *args):
        eps = self._epsilon * self._epsilonDecay
        eps = max(eps, 0.01)
        # eps = self._epsilon / (self._episode + 1)
        self._episode += 1
        self._epsilon = eps

    def train(self, state, action, newState, reward, done):
        intAction = self._gameActions.index(action)

        x = self._stateToTensor(state)
        q = self._model(x)
        q_target = q.clone()

        # if done:
        #     q_target[intAction] = reward
        # else:
        x_new = self._stateToTensor(newState)
        q_new = self._model(x_new)
        a_new = q_new.argmax()
        q_target[intAction] += self._alpha * (reward + self._gamma * q_new[a_new] - q_target[intAction])

        self._optimizer.zero_grad()
        loss = self._lossFnc(q_target, q)
        loss.backward()
        self._optimizer.step()

    def save(self):
        pass

    def _stateToTensor(self, state):
        # occupancy grid flattened
        # x = state["occupancy_grid"]
        # x = from_numpy(x.reshape(-1))
        # x = convert_image_dtype(x)

        # juste position tete normalisee avec direction food tout concatenee
        grid = state["occupancy_grid"]
        head_p = state["head_position"]
        head_d = state["head_direction"]
        food_d = state["food_position"] - state["head_position"]

        head_p = head_p / grid.shape[:2]
        food_d = food_d / norm(food_d)

        x = np.concatenate((head_p, head_d, food_d), dtype=np.float32)
        x = from_numpy(x)

        # print(type(x))
        # print(x.shape)
        # print(x)
        # print(state.keys())

        # state["head_position"] - state["food_position"]

        # print(  )
        # print(  )

        # exit(1)


        return x
