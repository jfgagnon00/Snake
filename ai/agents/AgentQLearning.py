import numpy as np

from collections import deque
from game import GameAction
from numpy.linalg import norm
from numpy.random import choice as np_choice
from random import random, choice as py_choice
from torch import argmax, from_numpy, no_grad
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

        numInputs = 10
        numOutput = len(GameAction)

        self._memory = deque(maxlen=50000)

        self._gamma = trainConfig.gamma
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay

        self._model = _QNet(numInputs,
                          [256],
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

        return GameAction(gameAction)

    def onEpisodeDone(self, *args):
        # do long memory train
        self._epsilon *= self._epsilonDecay

    def train(self, state, action, newState, reward, done):
        intAction = self._gameActions.index(action)

        x = self._stateToTensor(state)
        q = self._model(x)

        with no_grad():
            q_target = q.clone()
            if done:
                x_new = None
                q_target[intAction] = reward
            else:
                x_new = self._stateToTensor(newState)
                q_new = self._model(x_new)
                a_new = q_new.argmax()
                q_target[intAction] = reward + self._gamma * q_new[a_new]

        self._memory.append((x, intAction, x_new, reward, done))

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

        # juste position tete normalisee avec direction food tout concatene
        grid = state["occupancy_grid"]
        head_p = state["head_position"]
        head_d = state["head_direction"]
        food_d = state["food_position"] - state["head_position"]

        head_np = head_p / grid.shape[:2]
        food_nd = food_d / norm(food_d)

        head_ccw_d = np.array((-head_d[1], head_d[0]))

        col_head_d = self._first_collision(grid, head_p, head_d)
        col_ccw_d = self._first_collision(grid, head_p, head_ccw_d)
        col_cw_d = self._first_collision(grid, head_p, -head_ccw_d)

        # print()
        # print(grid.shape)
        # print(head_p, head_d)
        # print(col_head_d)
        # print(col_ccw_d)
        # print(col_cw_d)

        # # print(type(x))
        # # print(x.shape)
        # # print(x)
        # # print(state.keys())

        # print()
        # print()

        # exit(1)

        x = np.concatenate((head_d,
                            food_nd,
                            col_head_d,
                            col_ccw_d,
                            col_cw_d), dtype=np.float32)
        x = from_numpy(x)
        return x

    def _first_collision(self, grid, start, dir):
        c = np.copy(start)

        while True:
            if c[0] <= 0 or c[0] >= (grid.shape[0] - 1):
                break

            if c[1] <= 0 or c[1] >= (grid.shape[1] - 1):
                break

            if not np.array_equal(c, start) and grid[c[0], c[1], 0] != 0:
                break

            c = c + dir

        return (c - start) / grid.shape[:2]
