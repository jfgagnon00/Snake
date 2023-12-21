import numpy as np
import os

from collections import deque
from game import GameAction
from numpy.random import choice as np_choice
from random import random, sample as py_sample
from torch import from_numpy, \
                no_grad, \
                nonzero as torch_nonzero, \
                min as torch_min, \
                tensor, \
                save, \
                unsqueeze, \
                vstack, \
                int64 as torch_int64, \
                float32 as torch_float32
from torch.nn import Linear, Module, MSELoss, ReLU, Sequential, Conv2d, MaxPool2d, Flatten
from torch.optim import Adam
from torchvision.transforms.functional import convert_image_dtype
from torchsummary import summary

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

        self._net.append(Conv2d(1, 8, 5, padding='same'))
        self._net.append(ReLU())

        self._net.append(Conv2d(8, 16, 5, padding='same'))
        self._net.append(ReLU())

        self._net.append(Conv2d(16, 32, 5, padding='same'))
        self._net.append(ReLU())

        self._net.append(Flatten())

        self._net.append(Linear(1152, 512))
        self._net.append(ReLU())

        self._net.append(Linear(512, 256))
        self._net.append(ReLU())

        self._net.append(Linear(256, 128))
        self._net.append(ReLU())

        self._net.append(Linear(128, len(GameAction)))

    def forward(self, x):
        return self._net(x)

class Agent47(AgentBase):
    MEMORY_SIZE = 50000
    BATCH_SIZE = 2500

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        self._gamma = trainConfig.gamma
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay
        self._gameActions = list(GameAction)

        self._replayBuffer = deque(maxlen=Agent47.MEMORY_SIZE)

        # clipped DQN
        self._models = [self._buildModel(trainConfig.lr),
                        self._buildModel(trainConfig.lr)]

        if False:
            summary(self._models[0][0], (1, 6, 6))

    def getAction(self, state):
        if random() < self._epsilon:
            gameAction = np_choice(self._gameActions)
        else:
            x = self._stateToTensor(state)
            q = self._evalModel(0, x)
            intAction = q.argmax().item()
            gameAction = self._gameActions[intAction]

        return GameAction(gameAction)

    def onEpisodeBegin(self, episode, stats):
        stats.loc[0, "Epsilon"] = self._epsilon

    def onEpisodeDone(self, *args):
        self._epsilon *= self._epsilonDecay

        # entraine avec vieux samples en mode batch
        size = min(len(self._replayBuffer), Agent47.BATCH_SIZE)
        if size > 0:
            batch = py_sample(self._replayBuffer, size)
            states, intActions, newStates, rewards, dones = zip(*batch)
            self._train(vstack(states),
                        tensor(intActions, dtype=torch_int64).view(-1, 1),
                        vstack(newStates),
                        tensor(rewards, dtype=torch_float32),
                        tensor(dones, dtype=torch_float32))

    def train(self, state, action, newState, reward, done):
        # entraine avec chaque experience
        x = self._stateToTensor(state)
        intAction = self._gameActions.index(action)
        x_new = self._stateToTensor(newState)

        self._replayBuffer.append((x, intAction, x_new, reward, done))

        self._train(x,
                    tensor(intAction, dtype=torch_int64).view(-1, 1),
                    x_new,
                    tensor(reward, dtype=torch_float32),
                    tensor(done, dtype=torch_float32))

    def save(self, *args):
        path, filename = os.path.split(args[0])
        filename, _ = os.path.splitext(filename)

        os.makedirs(path, exist_ok=True)

        filename = os.path.join(path, f"{filename}.pth")
        save(self._models[0][0].state_dict(), filename)

    def _train(self, states, intActions, newStates, rewards, dones):
        # gather fait un lookup, donc enleve les dimensions
        q0 = self._evalModel(0, states).gather(1, intActions)
        q1 = self._evalModel(1, states).gather(1, intActions)

        with no_grad():
            q0_new = self._evalModel(0, newStates)
            q1_new = self._evalModel(1, newStates)
            q_new = torch_min(
                q0_new.max(dim=1)[0],
                q1_new.max(dim=1)[0]
            )

            q_target = rewards + self._gamma * (1 - dones) * q_new
            q_target = q_target.view(-1, 1)

        self._optimizeModel(0, q0, q_target)
        self._optimizeModel(1, q1, q_target)

    def _buildModel(self, lr):
        model = _LinearNet(10, [64], len(self._gameActions))
        return model, \
               Adam(model.parameters(), lr=lr), \
               MSELoss()

    def _evalModel(self, index, x):
        return self._models[index][0](x)

    def _optimizeModel(self, index, predicate, target):
        optimizer = self._models[index][1]
        loss = self._models[index][2]

        optimizer.zero_grad()
        loss(predicate, target).backward()
        optimizer.step()

    def _stateToTensor(self, state):
        if False:
            x = state["occupancy_grid"]

            # if self._gridStack is None:
            #     self._gridStack = x.copy()
            # else:
            #     self._gridStack = np.append(self._gridStack[:,:,-3:], x, axis=2)
            # x = from_numpy(self._gridStack)

            x = from_numpy(x)
            x = convert_image_dtype(x)
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
            x = from_numpy(x)

        return unsqueeze(x, 0)
