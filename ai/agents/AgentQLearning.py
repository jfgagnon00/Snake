import numpy as np
import os

from collections import deque
from game import GameAction
from numpy.linalg import norm
from numpy.random import choice as np_choice
from random import random, sample as py_sample
from torch import from_numpy, \
                no_grad, \
                nonzero as torch_nonzero, \
                tensor, \
                save, \
                unsqueeze, \
                vstack, \
                bool as torch_bool, \
                int32 as torch_int32, \
                float32 as torch_float32
from torch.nn import Linear, Module, MSELoss, ReLU, Sequential
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
    MEMORY_SIZE = 50000
    BATCH_SIZE = 2500

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        numInputs = 12
        numOutput = len(GameAction)

        self._memory = deque(maxlen=Agent47.MEMORY_SIZE)

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
        # reentraine avec vieux samples en mode batch
        if len(self._memory) > Agent47.BATCH_SIZE:
            batch = py_sample(self._memory, Agent47.BATCH_SIZE)
        else:
            batch = self._memory

        if len(batch) > 0:
            states, intActions, newStates, rewards, dones = zip(*batch)

            self._train(vstack(states),
                        tensor(intActions, dtype=torch_int32),
                        vstack(newStates),
                        unsqueeze(tensor(rewards, dtype=torch_float32), 0),
                        tensor(dones, dtype=torch_bool))

        self._epsilon *= self._epsilonDecay

    def train(self, state, action, newState, reward, done):
        # entraine avec chaque experience
        x = self._stateToTensor(state)
        intAction = self._gameActions.index(action)
        x_new = self._stateToTensor(newState)

        self._memory.append((x, intAction, x_new, reward, done))

        self._train(unsqueeze(x, 0),
                    tensor(intAction, dtype=torch_int32),
                    unsqueeze(x_new, 0),
                    unsqueeze(tensor(reward, dtype=torch_float32), 0),
                    tensor(done, dtype=torch_bool))

    def save(self, *args):
        path, filename = os.path.split(args[0])
        filename, _ = os.path.splitext(filename)

        os.makedirs(path, exist_ok=True)

        filename = os.path.join(path, f"{filename}.pth")
        save(self._model.state_dict(), filename)

    def _train(self, states, intActions, newStates, rewards, dones):
        q = self._model(states)

        with no_grad():
            q_new = self._model(newStates)
            q_new = q_new.max(dim=1)[0]

            done_indices = torch_nonzero(dones)
            q_new[done_indices] = 0

            q_target = q.clone()
            q_target[:, intActions] = rewards + self._gamma * q_new

        self._optimizer.zero_grad()
        loss = self._lossFnc(q_target, q)
        loss.backward()
        self._optimizer.step()

    def _stateToTensor(self, state):
        # occupancy grid flattened
        # x = state["occupancy_grid"]
        # x = from_numpy(x.reshape(-1))
        # x = convert_image_dtype(x)

        # positions/directions normalisees concatenees
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

        x = np.concatenate((head_np,
                            head_d,
                            food_nd,
                            col_head_d,
                            col_ccw_d,
                            col_cw_d), dtype=np.float32)
        return from_numpy(x)

    def _first_collision(self, grid, start, dir):
        # TODO: move dans simulation
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
