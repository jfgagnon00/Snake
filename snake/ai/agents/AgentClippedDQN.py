import numpy as np
import os

from random import random, choice
from torch import from_numpy, \
                no_grad, \
                min as torch_min, \
                maximum as torch_maximum, \
                tensor, \
                save, \
                load, \
                unsqueeze, \
                vstack, \
                int64 as torch_int64, \
                float32 as torch_float32
from torch.optim import Adam
from torchsummary import summary

from snake.game import GameAction, GridOccupancy
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.nets import _LinearNet, _ConvNet
from snake.ai.PriorityReplayBuffer import _PriorityReplayBuffer
from snake.ai.NStepPriorityReplayBuffer import _NStepPriorityReplayBuffer


class AgentClippedDQN(AgentBase):
    MEMORY_SIZE = 64_000
    BATCH_SIZE = 32

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        # misc parameters
        self._gameActions = list(GameAction)
        self._useConv = trainConfig.useConv

        # priority replay buffer
        self._replayBuffer = _NStepPriorityReplayBuffer(AgentClippedDQN.MEMORY_SIZE,
            trainConfig.alpha,
            trainConfig.beta,
            trainConfig.betaAnnealingSteps,
            trainConfig.gamma,
            trainConfig.nStep)

        # clipped DQN
        self._gamma = trainConfig.gamma ** trainConfig.nStep
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay
        self._epsilonMin = trainConfig.epsilonMin
        self._models = [self._buildModel(trainConfig,
                                         simulationConfig.gridWidth,
                                         simulationConfig.gridHeight),
                        self._buildModel(trainConfig,
                                         simulationConfig.gridWidth,
                                         simulationConfig.gridHeight)]

        self._useFrameStack = trainConfig.useFrameStack

        if False:
            summary(self._models[0][0], (1, 3, 6, 6))
            exit(-1)

    def getAction(self, state):
        if random() < self._epsilon:
            gameAction = np.random.choice(self._gameActions)
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
        self._epsilon = max(self._epsilon, self._epsilonMin)

    def train(self, state, action, newState, reward, done):
        self._replayBuffer.append(self._stateToTensor(state),
                                  self._gameActions.index(action),
                                  self._stateToTensor(newState),
                                  reward,
                                  done)
        self._trainBatch()

    def save(self, *args):
        path, filename = os.path.split(args[0])
        filename, _ = os.path.splitext(filename)

        os.makedirs(path, exist_ok=True)

        file = os.path.join(path, f"{filename}-0.pth")
        self._save(file, 0)

        file = os.path.join(path, f"{filename}-1.pth")
        self._save(file, 1)

    def load(self, *args):
        filename = args[0]

        self._load(f"{filename}-0.pth", 0)
        self._load(f"{filename}-1.pth", 1)

    def _save(self, filename, index):
        data = {"model": self._models[index][0].state_dict(),
                "optimizer": self._models[index][1].state_dict()}
        save(data, filename)

    def _load(self, filename, index):
        states = load(filename)
        self._models[index][0].load_state_dict(states["model"])
        self._models[index][1].load_state_dict(states["optimizer"])

    def _trainBatch(self):
        replaySize = len(self._replayBuffer)

        if replaySize >= AgentClippedDQN.BATCH_SIZE:
            samples = self._replayBuffer.sample(AgentClippedDQN.BATCH_SIZE)
            states, intActions, newStates, rewards, dones, weights, indices = samples
            errors = self._train(vstack(states),
                                 tensor(intActions, dtype=torch_int64).view(-1, 1),
                                 vstack(newStates),
                                 tensor(rewards, dtype=torch_float32),
                                 tensor(dones, dtype=torch_float32),
                                 tensor(weights, dtype=torch_float32))
            self._replayBuffer.updatePriorities(indices, errors)

    def _train(self, states, intActions, newStates, rewards, dones, weights=None):
        with no_grad():
            q0_new = self._evalModel(0, newStates)
            q1_new = self._evalModel(1, newStates)
            q_new = torch_min(
                q0_new.max(dim=1)[0],
                q1_new.max(dim=1)[0]
            )

            q_target = rewards + self._gamma * (1 - dones) * q_new
            q_target = q_target.view(-1, 1)

        # gather fait un lookup, donc enleve les dimensions
        q0 = self._evalModel(0, states).gather(1, intActions)
        q1 = self._evalModel(1, states).gather(1, intActions)

        e0 = self._optimizeModel(0, q0, q_target, weights)
        e1 = self._optimizeModel(1, q1, q_target, weights)

        return torch_maximum(e0, e1).detach().numpy() + 1e-6

    def _buildModel(self, trainConfig, width, height):
        if self._useConv:
            numInputs = trainConfig.frameStack if trainConfig.useFrameStack else 1
            numInputs *= 3
            model = _ConvNet(width, height, numInputs, len(self._gameActions))
        else:
            model = _LinearNet(width * height * 3 + 4, [512], len(self._gameActions))

        model.train()

        return model, Adam(model.parameters(), lr=trainConfig.lr)

    def _evalModel(self, index, x):
        return self._models[index][0](x)

    def _optimizeModel(self, index, predicate, target, weights=None):
        optimizer = self._models[index][1]

        optimizer.zero_grad()
        error = loss = (predicate - target) ** 2
        if not weights is None:
            loss = loss * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        return error

    def _stateToTensor(self, state):
        grid = state["occupancy_grid"]
        grid = np.squeeze(grid)

        if self._useFrameStack:
            gg = np.zeros((grid.shape[0], 3, *grid.shape[1:]), dtype=grid.dtype)

            for i in range(grid.shape[0]):
                gg[i, 0] = np.where(grid[i,:,:] == GridOccupancy.SNAKE_BODY, 1, 0)
                gg[i, 1] = np.where(grid[i,:,:] == GridOccupancy.SNAKE_HEAD, 1, 0)
                gg[i, 2] = np.where(grid[i,:,:] == GridOccupancy.FOOD, 1, 0)

            gg = gg.reshape((-1, *grid.shape[1:]))
        else:
            gg = np.zeros((3, *grid.shape), dtype=grid.dtype)
            gg[0] = np.where(grid == GridOccupancy.SNAKE_BODY, 1, 0)
            gg[1] = np.where(grid == GridOccupancy.SNAKE_HEAD, 1, 0)
            gg[2] = np.where(grid == GridOccupancy.FOOD, 1, 0)

        if self._useConv:
            x = from_numpy(gg.astype(np.float32))
        else:
            head_p = state["head_position"]
            food_p = state["food_position"]
            food_d = food_p - head_p

            n = food_d[0] < 0
            s = food_d[0] > 0
            w = food_d[1] < 0
            e = food_d[1] > 0

            gg = gg.flatten()
            gg = np.append(gg, [n, s, w, e])

            x = from_numpy(gg.astype(np.float32))

        return unsqueeze(x, 0)
