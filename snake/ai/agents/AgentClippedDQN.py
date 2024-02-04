import numpy as np
import os

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
from torch.nn.functional import softmax
from torchsummary import summary

from snake.core import Vector
from snake.game import GameAction, GameDirection, GridOccupancy
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.nets import _ConvNet, _DuelingConvNet, _LinearNet
from snake.ai.ReplayBuffer import _ReplayBuffer
from snake.ai.PriorityReplayBuffer import _PriorityReplayBuffer
from snake.ai.NStepPriorityReplayBuffer import _NStepPriorityReplayBuffer


class AgentClippedDQN(AgentBase):
    MEMORY_SIZE = 8_192
    BATCH_SIZE = 64

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        # misc parameters
        self._gameActions = list(GameAction)
        self._useConv = trainConfig.useConv
        self._gridCenter = Vector(simulationConfig.gridWidth,
                                  simulationConfig.gridHeight).scale(0.5) - Vector(0.5, 0.5)

        # replay buffer
        self._replayBuffer = _ReplayBuffer(AgentClippedDQN.MEMORY_SIZE)

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
            summary(self._models[0][0], \
                    (1, simulationConfig.gridHeight, simulationConfig.gridWidth))
            exit(-1)

    def getAction(self, state):
        self._NumActions += 1

        x0, x1, actionFlags = self._stateToTensor(state)

        if np.random.uniform() < self._epsilon:
            self._NumRandomActions += 1
            self._lastQvalues = np.zeros((len(GameAction)), dtype=np.float32)
            self._lastActionProbs = np.array(actionFlags, dtype=np.float32)
            self._lastActionProbs = self._lastActionProbs / (self._lastActionProbs.sum() + 1e-6)

            intAction = np.random.choice(len(GameAction), p=self._lastActionProbs)
        else:
            with no_grad():
                q = self._evalModel(0, x0, x1)
                self._lastQvalues = q.numpy().flatten()
                self._lastActionProbs = softmax(q, dim=1).numpy().flatten()

                # intAction = q.argmax().item()
                intAction = np.random.choice(len(GameAction), p=self._lastActionProbs)

        return self._gameActions[intAction]

    def onEpisodeBegin(self, episode, stats):
        stats.loc[0, "Epsilon"] = self._epsilon
        self._trainLoss = np.zeros((1), dtype=np.float32)
        self._NumRandomActions = 0
        self._NumActions = 0

    def onEpisodeDone(self, episode, stats):
        self._epsilon *= self._epsilonDecay
        self._epsilon = max(self._epsilon, self._epsilonMin)
        stats.loc[0, "TrainLossMin"] = self._trainLoss.min()
        stats.loc[0, "TrainLossMax"] = self._trainLoss.max()
        stats.loc[0, "TrainLossMean"] = self._trainLoss.mean()

        for i, a in enumerate(self._gameActions):
            stats.loc[0, f"Q_{a.name}"] = self._lastQvalues[i]

        for i, a in enumerate(self._gameActions):
            stats.loc[0, f"P_{a.name}"] = self._lastActionProbs[i]

        stats.loc[0, f"RandomActionsRatio"] = self._NumRandomActions / self._NumActions

    def train(self, state, action, newState, reward, done):
        self._replayBuffer.append(self._stateToTensor(state),
                                  self._gameActions.index(action),
                                  self._stateToTensor(newState),
                                  reward,
                                  done)
        self._trainBatch()

    def save(self, *args):
        if len(args) > 0:
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
        data = {"epsilon": self._epsilon,
                "model": self._models[index][0].state_dict(),
                "optimizer": self._models[index][1].state_dict()}
        save(data, filename)

    def _load(self, filename, index):
        states = load(filename)
        self._epsilon = states["epsilon"]
        self._models[index][0].load_state_dict(states["model"])
        self._models[index][1].load_state_dict(states["optimizer"])

    def _trainBatch(self):
        replaySize = len(self._replayBuffer)

        if replaySize >= AgentClippedDQN.BATCH_SIZE:
            samples = self._replayBuffer.sample(AgentClippedDQN.BATCH_SIZE)
            states, intActions, newStates, rewards, dones = samples

            def unpack(states):
                x0 = []
                x1 = []
                for s in states:
                    x0.append(s[0])
                    x1.append(s[1])
                return vstack(x0), vstack(x1)

            errors = self._train(unpack(states),
                                 tensor(intActions, dtype=torch_int64).view(-1, 1),
                                 unpack(newStates),
                                 tensor(rewards, dtype=torch_float32),
                                 tensor(dones, dtype=torch_float32))
            self._trainLoss = np.append(self._trainLoss, errors.mean())

    def _train(self, states, intActions, newStates, rewards, dones, weights=None):
        self._models[0][0].train()
        self._models[1][0].train()

        with no_grad():
            q0_new = self._evalModel(0, *newStates)
            q1_new = self._evalModel(1, *newStates)
            q_new = torch_min(
                q0_new.max(dim=1)[0],
                q1_new.max(dim=1)[0]
            )

            q_target = rewards + self._gamma * (1 - dones) * q_new
            q_target = q_target.view(-1, 1)

        # gather fait un lookup, donc enleve les dimensions
        q0 = self._evalModel(0, *states).gather(1, intActions)
        q1 = self._evalModel(1, *states).gather(1, intActions)

        e0 = self._optimizeModel(0, q0, q_target, weights)
        e1 = self._optimizeModel(1, q1, q_target, weights)

        e0 = e0.detach().numpy()

        self._models[0][0].eval()
        self._models[1][0].eval()

        return e0

    def _buildModel(self, trainConfig, width, height):
        if self._useConv:
            assert not trainConfig.useFrameStack, "Frame stack non supporte"
            # model = _ConvNet(width, height, 3, 6, len(self._gameActions))
            model = _DuelingConvNet(width, height, 3, 6, len(self._gameActions))
        else:
            # numFrames = trainConfig.frameStack if trainConfig.useFrameStack else 1
            # miscs = 0 if trainConfig.useFrameStack else 4
            model = _LinearNet(width * height * 3 + 6, [256, 256], len(self._gameActions))

        model.eval()

        return model, Adam(model.parameters(), lr=trainConfig.lr)

    def _evalModel(self, index, *args):
        return self._models[index][0](*args)

    def _optimizeModel(self, index, predicate, target, weights=None):
        error = (predicate - target) ** 2
        if not weights is None:
            error = error * weights
        loss = error.mean()

        optimizer = self._models[index][1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return error

    def _stateToTensor(self, state):
        _, w = state["occupancy_grid"].shape[1:]

        grid, head_p, food_p = self._applySymmetry(state)
        grid = self._splitOccupancyGrid(grid, pad=False)

        if not food_p is None:
            food_d = food_p - head_p

            food_cw = 1 if food_d.x > 0 else 0
            food_ccw = 1 if food_d.x < 0 else 0
            food_f = 1 if food_d.y < 0 else 0

            head_cw = 1 if head_p.x < (w - 1) else 0
            head_ccw = 1 if head_p.x > 0 else 0
            head_f = 1 if head_p.y > 0 else 0
        else:
            food_cw = 0
            food_ccw = 0
            food_f = 0

            head_cw = 0
            head_ccw = 0
            head_f = 0

        flags = np.array([food_cw, food_ccw, food_f, head_cw, head_ccw, head_f])

        x0 = from_numpy(grid.astype(np.float32))
        x1 = from_numpy(flags.astype(np.float32))

        return unsqueeze(x0, 0), unsqueeze(x1, 0), (head_cw, head_ccw, head_f)

    def _applySymmetry(self, state):
        head_d = state["head_direction"]
        head_d = Vector.fromNumpy(head_d)

        # simplifier state: toujours mettre par rapport a NORTH
        head_d = GameDirection(head_d)
        if head_d == GameDirection.EAST:
            # CCW
            k = 1
        elif head_d == GameDirection.WEST:
            # CW
            k = -1
        elif head_d == GameDirection.SOUTH:
            # 180 degrees
            k = 2
        else:
            k = 0

        grid = state["occupancy_grid"]
        grid = np.rot90(grid, k=k, axes=(1, 2))

        head_p = state["head_position"]
        head_p = Vector.fromNumpy(head_p) - self._gridCenter
        head_p = head_p.rot90(k) + self._gridCenter

        food_p = state["food_position"]
        if not food_p is None:
            food_p = Vector.fromNumpy(food_p) - self._gridCenter
            food_p = food_p.rot90(k) + self._gridCenter

        return grid, head_p, food_p

    def _splitOccupancyGrid(self, occupancyGrid, pad=False):
        if pad:
            # pad le pourtour pour avoir obstacle
            occupancyGrid = np.pad(occupancyGrid,
                                    ((0, 0), (1, 1), (1, 1)),
                                    constant_values=GridOccupancy.SNAKE_BODY)

        shape = (3, *occupancyGrid.shape[1:])

        occupancyStack = np.zeros(shape=shape, dtype=np.int32)
        occupancyStack[0] = np.where(occupancyGrid[0,:,:] == GridOccupancy.SNAKE_BODY, 1, 0)
        occupancyStack[1] = np.where(occupancyGrid[0,:,:] == GridOccupancy.SNAKE_HEAD, 1, 0)
        occupancyStack[2] = np.where(occupancyGrid[0,:,:] == GridOccupancy.FOOD, 1, 0)

        return occupancyStack
