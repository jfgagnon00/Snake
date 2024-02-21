import numpy as np
import os

from torch import from_numpy, \
                no_grad, \
                min as torch_min, \
                tensor, \
                save, \
                load, \
                unsqueeze, \
                vstack, \
                int64 as torch_int64, \
                float32 as torch_float32
from torch.optim import Adam
from torchsummary import summary

from snake.core import Vector
from snake.game import GameAction
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.nets import _ConvNet, _DuelingConvNet, _LinearNet
from snake.ai.ReplayBuffer import _ReplayBuffer
from snake.ai.NStepReplayBuffer import _NStepReplayBuffer
from snake.ai.StateProcessor import _StateProcessor


class AgentClippedDQN(AgentBase):
    MEMORY_SIZE = 8_192
    BATCH_SIZE = 64

    def __init__(self, trainConfig, simulationConfig) -> None:
        super().__init__()

        # misc parameters
        self._numGameActions = len(GameAction)
        self._gameActions = list(GameAction)
        self._useConv = trainConfig.useConv
        self._gridCenter = Vector(simulationConfig.gridWidth,
                                  simulationConfig.gridHeight).scale(0.5) - Vector(0.5, 0.5)

        # replay buffer
        self._replayBuffer = _NStepReplayBuffer(AgentClippedDQN.MEMORY_SIZE,
                                                trainConfig.gamma,
                                                trainConfig.nStep)

        # clipped DQN
        self._gamma = trainConfig.gamma ** trainConfig.nStep
        self._epsilon = trainConfig.epsilon
        self._epsilonDecay = trainConfig.epsilonDecay
        self._epsilonMin = trainConfig.epsilonMin
        self._tau = trainConfig.tau
        self._models = [self._buildModel(trainConfig,
                                         simulationConfig.gridWidth,
                                         simulationConfig.gridHeight),
                        self._buildModel(trainConfig,
                                         simulationConfig.gridWidth,
                                         simulationConfig.gridHeight,
                                         False)]
        self._copyWeights()

        self._stateProcessor = _StateProcessor()

        if False:
            summary(self._models[0][0], \
                    (1, 1, simulationConfig.gridHeight, simulationConfig.gridWidth))
            exit(-1)

    def getAction(self, state):
        self._NumActions += 1

        x0, x1, actionFlags = self._stateProcessing(state)

        self._lastActionFlags = actionFlags
        actionFlags = np.array(actionFlags, dtype=np.float32)
        actionsAvailable = np.nonzero(actionFlags)[0]

        if np.random.uniform() < self._epsilon:
            self._NumRandomActions += 1
            self._lastActionRandom = 1
            self._lastQvalues = actionFlags.copy()
            lastActionProbs = actionFlags.copy()
        else:
            self._lastActionRandom = 0
            with no_grad():
                q = self._evalModel(0, x0, x1)
                self._lastQvalues = q.numpy().flatten().copy()
                lastActionProbs = self._lastQvalues.copy()

        # transformer valeurs en probabilites
        sum_ = 0
        if len(actionsAvailable) > 0:
            lastActionProbs = lastActionProbs[actionsAvailable]
            lastActionProbs -= lastActionProbs.max()
            lastActionProbs = np.exp(lastActionProbs)
            lastActionProbs = np.nan_to_num(lastActionProbs)
            sum_ = lastActionProbs.sum()

        if sum_ < 1e-6:
            actionsAvailable = np.arange(self._numGameActions)
            lastActionProbs = np.ones(self._numGameActions, dtype=np.float32)
            sum_ = lastActionProbs.sum()

        lastActionProbs /= sum_

        self._lastActionProbs = np.ones(self._numGameActions, dtype=np.float32)
        self._lastActionProbs[actionsAvailable] = lastActionProbs

        intAction = np.random.choice(actionsAvailable, p=lastActionProbs)
        return self._gameActions[intAction]

    def onEpisodeBegin(self, episode, frameStats):
        frameStats.loc[0, "Epsilon"] = self._epsilon
        self._trainLoss = np.zeros((1), dtype=np.float32)
        self._NumRandomActions = 0
        self._NumActions = 0
        self._lastActionFlags = [9, 9, 9]
        self._lastActionRandom = 0

    def onEpisodeDone(self, episode, frameStats):
        self._epsilon *= self._epsilonDecay
        self._epsilon = max(self._epsilon, self._epsilonMin)
        frameStats.loc[0, "TrainLossMin"] = self._trainLoss.min()
        frameStats.loc[0, "TrainLossMax"] = self._trainLoss.max()
        frameStats.loc[0, "TrainLossMean"] = self._trainLoss.mean()

        for i, a in enumerate(self._gameActions):
            frameStats.loc[0, f"Q_{a.name}"] = self._lastQvalues[i]

        for i, a in enumerate(self._gameActions):
            frameStats.loc[0, f"P_{a.name}"] = self._lastActionProbs[i]

        for i, a in enumerate(self._gameActions):
            frameStats.loc[0, f"Flag_{a.name}"] = self._lastActionFlags[i]

        frameStats.loc[0, f"LastActionRandom"] = self._lastActionRandom
        frameStats.loc[0, f"RandomActionsRatio"] = self._NumRandomActions / self._NumActions

    def train(self, state, action, newState, reward, done):
        self._replayBuffer.append(self._stateProcessing(state),
                                  self._gameActions.index(action),
                                  self._stateProcessing(newState),
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
                "model": self._models[index][0].state_dict()}

        optimizer = self._models[index][1]
        if not optimizer is None:
            data["optimizer"] = optimizer.state_dict()

        save(data, filename)

    def _load(self, filename, index):
        states = load(filename)
        self._epsilon = states["epsilon"]
        self._models[index][0].load_state_dict(states["model"])

        if "optimizer" in states:
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

                    if not s[1] is None:
                        x1.append(s[1])

                x0 = vstack(x0)
                x1 = None if len(x1) == 0 else vstack(x1)

                return x0, x1

            loss = self._train(unpack(states),
                               tensor(intActions, dtype=torch_int64).view(-1, 1),
                               unpack(newStates),
                               tensor(rewards, dtype=torch_float32),
                               tensor(dones, dtype=torch_float32))
            self._trainLoss = np.append(self._trainLoss, loss)

    def _train(self, states, intActions, newStates, rewards, dones, weights=None):
        with no_grad():
            q0_new = self._evalModel(0, *newStates)
            q1_new = self._evalModel(1, *newStates)
            q_new = torch_min(
                q0_new.max(dim=1)[0],
                q1_new.max(dim=1)[0]
            )
            q_target = rewards + self._gamma * (1 - dones) * q_new
            q_target = q_target.view(-1, 1)

        self._models[0][0].train()

        # gather fait un lookup, donc enleve les dimensions
        q = self._evalModel(0, *states).gather(dim=1, index=intActions)
        loss = self._optimizeModel(0, q, q_target, weights)

        self._models[0][0].eval()
        self._copyWeights(self._tau)

        return loss

    def _buildModel(self, trainConfig, width, height, optimizer=True):
        numImputs = 0

        if self._useConv:
            model = _ConvNet(width, height, 3, numImputs, len(self._gameActions))
            # model = _DuelingConvNet(width, height, 3, numImputs, len(self._gameActions))
        else:
            model = _LinearNet(width * height * 3 + numImputs, [256, 256], len(self._gameActions))

        model.eval()
        optimizer = Adam(model.parameters(), lr=trainConfig.lr) if optimizer else None

        return model, optimizer

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

        return loss.detach().numpy()

    def _copyWeights(self, tau=None):
        p0 = self._models[0][0].state_dict()
        if tau is None:
            self._models[1][0].load_state_dict(p0)
        else:
            p1 = self._models[0][0].state_dict()
            for key in p0:
                p1[key] = tau * p0[key] + (1 - tau) * p1[key]
            self._models[1][0].load_state_dict(p1)

    def _stateProcessing(self, state):
        tensors = self._stateProcessor(state)
        return self._stateToTorch(*tensors)

    def _stateToTorch(self, x0, x1, head_flags):
        x0 = from_numpy(x0.astype(np.float32))
        x0 = unsqueeze(x0, 0)

        if not x1 is None:
            x1 = from_numpy(x1.astype(np.float32))
            x1 = unsqueeze(x1, 0)

        return x0, x1, head_flags

