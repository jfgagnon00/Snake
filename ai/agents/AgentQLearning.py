import numpy as np
import os

from collections import deque
from game import GameAction
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
from torch.nn import Linear, Module, MSELoss, ReLU, Sequential, Conv2d
from torch.optim import Adam
from torchvision.transforms.functional import convert_image_dtype

from .AgentBase import AgentBase


class _LinearNet(Module):
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

class _ConvNet(Module):
    def __init__(self):
        super().__init__()

        self._net = Sequential()

        # self._net.append(Conv2d(prevSize, size))
        # self._net.append(ReLU())

        self._net.append(Linear(2, 2))

    def forward(self, x):
        return self._net(x)

class Agent47(AgentBase):
    MEMORY_SIZE = 50000
    BATCH_SIZE = 2500

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        self._memory = deque(maxlen=Agent47.MEMORY_SIZE)

        self._gamma = trainConfig.gamma
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay

        self._gridStack = None
        self._model = _LinearNet(10,
                                 [64],
                                 len(GameAction))

        self._optimizer = Adam(self._model.parameters(), lr=trainConfig.lr)
        self._lossFnc = MSELoss()
        self._gameActions = list(GameAction)

    def reset(self):
        self._gridStack = None

    def getAction(self, state):
        if random() < self._epsilon:
            gameAction = np_choice(self._gameActions)
        else:
            x = self._stateToTensor(state)
            actions = self._model(x)
            intAction = actions.argmax().item()
            gameAction = self._gameActions[intAction]

        return GameAction(gameAction)

    def onEpisodeBegin(self, episode, stats):
        stats.loc[0, "Epsilon"] = self._epsilon

    def onEpisodeDone(self, *args):
        # entraine avec vieux samples en mode batch
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
        if False:
            x = state["occupancy_grid"]

            if self._gridStack is None:
                self._gridStack = x.copy()
            else:
                self._gridStack = np.append(self._gridStack[:,:,-3:], x, axis=2)

            x = from_numpy(self._gridStack)
            x = convert_image_dtype(x)

            return x
        else:
            # positions/directions normalisees concatenees
            grid = state["occupancy_grid"]
            grid_size = grid.shape[1:]

            head_d = state["head_direction"]

            food_d = state["food_position"] - state["head_position"]
            food_d[0] = np.sign(food_d[0])
            food_d[1] = np.sign(food_d[1])

            col_forward = state["collision_forward"] / grid_size
            col_ccw = state["collision_ccw"] / grid_size
            col_cw = state["collision_cw"] / grid_size

            x = np.concatenate((head_d,
                                food_d,
                                col_forward,
                                col_ccw,
                                col_cw),
                                dtype=np.float32)
            return from_numpy(x)
