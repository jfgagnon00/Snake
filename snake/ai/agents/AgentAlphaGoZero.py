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
from snake.ai.nets import _ConvNet
from snake.ai.ReplayBuffer import _ReplayBuffer
from snake.ai.StateProcessor import _StateProcessor


class AgentAlphaGoZero(AgentBase):
    MEMORY_SIZE = 8_192
    BATCH_SIZE = 64

    def __init__(self, configs):
        super().__init__()

        trainConfig = configs.train
        simulationConfig = configs.simulation

        # misc parameters
        self._numGameActions = len(GameAction)
        self._gameActions = list(GameAction)

        # replay buffer
        self._replayBuffer = _ReplayBuffer(AgentAlphaGoZero.MEMORY_SIZE)

        # clipped DQN
        self._model, self._optimizer = self._buildModel(trainConfig,
                                                        simulationConfig.gridWidth,
                                                        simulationConfig.gridHeight)

        self._stateProcessor = _StateProcessor()

        if False:
            summary(self._model,
                    (3, simulationConfig.gridHeight + 2, simulationConfig.gridWidth + 2),
                    None)
            exit(-1)

    def getAction(self, state):
        _, _, actionFlags = self._stateProcessing(state)

        actionFlags = np.array(actionFlags, dtype=np.float32)
        actionsAvailable = np.nonzero(actionFlags)[0]

        intAction = np.random.choice(actionsAvailable)
        return self._gameActions[intAction]

    def onEpisodeBegin(self, episode, frameStats):
        pass

    def onEpisodeDone(self, episode, frameStats):
        pass

    def train(self, state, info, action, newState, newInfo, reward, done):
        # self._replayBuffer.append(self._stateProcessing(state),
        #                           self._gameActions.index(action),
        #                           self._stateProcessing(newState),
        #                           reward,
        #                           done)
        # self._trainFromReplayBuffer()
        pass

    def save(self, *args):
        if len(args) > 0:
            path, filename = os.path.split(args[0])
            filename, _ = os.path.splitext(filename)

            os.makedirs(path, exist_ok=True)

            file = os.path.join(path, f"{filename}.pth")
            self._save(file)

    def load(self, *args):
        filename = args[0]
        self._load(f"{filename}.pth")

    def _save(self, filename):
        data = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        save(data, filename)

    def _load(self, filename):
        states = load(filename)
        self._model.load_state_dict(states["model"])
        self._optimizer.load_state_dict(states["optimizer"])

    def _trainFromReplayBuffer(self):
        # replaySize = len(self._replayBuffer)

        # if replaySize >= AgentAlphaGoZero.BATCH_SIZE:
        #     samples = self._replayBuffer.sample(AgentAlphaGoZero.BATCH_SIZE)
        #     states, intActions, newStates, rewards, dones = samples

        #     def unpack(states):
        #         x0 = []
        #         x1 = []
        #         for s in states:
        #             x0.append(s[0])

        #             if not s[1] is None:
        #                 x1.append(s[1])

        #         x0 = vstack(x0)
        #         x1 = None if len(x1) == 0 else vstack(x1)

        #         return x0, x1

        #     loss = self._trainBatch(unpack(states),
        #                             tensor(intActions, dtype=torch_int64).view(-1, 1),
        #                             unpack(newStates),
        #                             tensor(rewards, dtype=torch_float32),
        #                             tensor(dones, dtype=torch_float32))
        #     self._trainLoss = np.append(self._trainLoss, loss)
        pass

    def _trainBatch(self, states, intActions, newStates, rewards, dones, weights=None):
        pass

    def _buildModel(self, trainConfig, width, height, optimizer=True):
        numInputs = 0
        numChannels = 3

        model = _ConvNet(width, height, numChannels, numInputs, len(self._gameActions))
        # model = _DuelingConvNet(width, height, numChannels, numInputs, len(self._gameActions))

        model.eval()
        optimizer = Adam(model.parameters(), lr=trainConfig.lr) if optimizer else None

        return model, optimizer

    def _stateProcessing(self, state):
        stateProcessed = self._stateProcessor(state)
        return self._stateToTorch(*stateProcessed)

    def _stateToTorch(self, x0, x1, head_flags):
        x0 = from_numpy(x0.astype(np.float32))
        x0 = unsqueeze(x0, 0)

        if not x1 is None:
            x1 = from_numpy(x1.astype(np.float32))
            x1 = unsqueeze(x1, 0)

        return x0, x1, head_flags
