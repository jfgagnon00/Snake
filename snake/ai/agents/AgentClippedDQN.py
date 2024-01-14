import numpy as np
import os

from collections import deque
from numpy.random import choice as np_choice
from random import random
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

from snake.game import GameAction
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.nets import _LinearNet, _ConvNet


class AgentClippedDQN(AgentBase):
    MEMORY_SIZE = 32_000
    BATCH_SIZE = 64

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        # misc parameters
        self._gameActions = list(GameAction)
        self._useConv = trainConfig.useConv

        # priority replay buffer
        self._alpha = trainConfig.alpha
        self._beta = trainConfig.beta
        self._betaAnnealingSteps = trainConfig.betaAnnealingSteps
        self._replayBuffer = deque(maxlen=AgentClippedDQN.MEMORY_SIZE)
        self._replayBufferPriority = deque(maxlen=AgentClippedDQN.MEMORY_SIZE)
        self._maxPriority = 1

        # clipped DQN
        self._gamma = trainConfig.gamma
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay
        self._models = [self._buildModel(trainConfig.lr,
                                         simulationConfig.gridWidth,
                                         simulationConfig.gridHeight),
                        self._buildModel(trainConfig.lr,
                                         simulationConfig.gridWidth,
                                         simulationConfig.gridHeight)]

        if False:
            summary(self._models[0][0], (3, 6, 6))
            exit(-1)

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

    def train(self, state, action, newState, reward, done):
        x = self._stateToTensor(state)
        intAction = self._gameActions.index(action)
        x_new = self._stateToTensor(newState)

        self._replayBuffer.append((x, intAction, x_new, reward, done))
        self._replayBufferPriority.append(self._maxPriority)

        self._trainBatch()

    def save(self, *args):
        path, filename = os.path.split(args[0])
        filename, _ = os.path.splitext(filename)

        os.makedirs(path, exist_ok=True)

        file = os.path.join(path, f"{filename}-0.pth")
        save(self._models[0][0].state_dict(), file)

        file = os.path.join(path, f"{filename}-1.pth")
        save(self._models[1][0].state_dict(), file)

    def load(self, *args):
        filename = args[0]

        states = load(f"{filename}-0.pth")
        states = {k:v for k, v in states.items() if "_net.0" in k}
        self._models[0][0].load_state_dict(states, strict=False)

        states = load(f"{filename}-1.pth")
        states = {k:v for k, v in states.items() if "_net.0" in k}
        self._models[1][0].load_state_dict(states, strict=False)

    def _trainBatch(self):
        assert len(self._replayBuffer) == len(self._replayBufferPriority)

        replaySize = len(self._replayBuffer)

        if replaySize >= AgentClippedDQN.BATCH_SIZE:
            # TODO: implementation tres lente, a refaire
            props = np.array(self._replayBufferPriority) ** self._alpha
            props = props / props.sum()

            batchIndices = np_choice(replaySize, AgentClippedDQN.BATCH_SIZE, p=props)

            batch = [self._replayBuffer[i] for i in batchIndices]
            states, intActions, newStates, rewards, dones = zip(*batch)

            beta = self._beta + (1.0 - self._beta) * replaySize / self._betaAnnealingSteps
            beta = min(1.0, beta)
            weights = (replaySize * props[batchIndices]) ** -beta
            weights = weights / weights.max()

            errors = self._train(vstack(states),
                                tensor(intActions, dtype=torch_int64).view(-1, 1),
                                vstack(newStates),
                                tensor(rewards, dtype=torch_float32),
                                tensor(dones, dtype=torch_float32),
                                tensor(weights, dtype=torch_float32))

            for i, j in enumerate(batchIndices):
                error = errors[i, 0]
                self._maxPriority = max(error, self._maxPriority)
                self._replayBufferPriority[j] = error

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

    def _buildModel(self, lr, width, height):
        if self._useConv:
            model = _ConvNet(width, height, len(self._gameActions))
        else:
            model = _LinearNet(114, [512, 512], len(self._gameActions))

        model.train()

        return model, Adam(model.parameters(), lr=lr)

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
        if self._useConv:
            grid = state["occupancy_stack"]
            x = from_numpy(grid.astype(np.float32))
        elif False:
            grid = state["occupancy_stack"]
            x = from_numpy( grid.flatten().astype(np.float32) )
        else:
            grid = state["occupancy_grid"]
            grid_size = grid.shape[1:]

            head_p = state["head_position"] / grid_size
            length = float(state["length"])

            head_d = state["head_direction"]
            head_forward = 1 if head_d[1] > 0 else 0
            head_ccw = 1 if head_d[0] > 0 else 0
            head_cw = 1 if head_d[0] < 0 else 0

            if state["food_position"] is None:
                food_forward = 0
                food_ccw = 0
                food_cw = 0
            else:
                food_d = state["food_position"] - state["head_position"]
                food_forward = 1 if food_d[1] > 0 else 0
                food_ccw = 1 if food_d[0] > 0 else 0
                food_cw = 1 if food_d[0] < 0 else 0

            # col_forward = state["collision_forward"] / grid_size
            # col_ccw = state["collision_ccw"] / grid_size
            # col_cw = state["collision_cw"] / grid_size

            x = np.array((head_forward, head_ccw, head_cw,
                          food_forward, food_ccw, food_cw),
                        #   norm(col_forward),
                        #   norm(col_ccw),
                        #   norm(col_cw)),
                          dtype=np.float32)

            stack = state["occupancy_stack"]
            x = np.concatenate((x, stack.flatten().astype(np.float32)))
            x = from_numpy(x)

        return unsqueeze(x, 0)
