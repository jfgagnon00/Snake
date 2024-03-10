import numpy as np
import os

from itertools import islice
from torch import from_numpy, \
                no_grad, \
                tensor, \
                save, \
                load, \
                unsqueeze, \
                vstack, \
                int64 as torch_int64, \
                float32 as torch_float32
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torchsummary import summary

from snake.game import GameAction
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.mcts import _Mcts
from snake.ai.nets import _AlphaGoZeroConvNet
from snake.ai.ReplayBuffer import _ReplayBuffer
from snake.ai.StateProcessor import _StateProcessor


class AgentAlphaGoZero(AgentBase):
    MEMORY_SIZE = 8_192 * 5
    BATCH_SIZE = 64

    def __init__(self, configs):
        super().__init__()

        trainConfig = configs.train
        simulationConfig = configs.simulation

        # misc parameters
        self._numGameActions = len(GameAction)
        self._gameActions = list(GameAction)
        self._stateProcessor = _StateProcessor()

        # replay buffer
        self._replayBuffer = _ReplayBuffer(AgentAlphaGoZero.MEMORY_SIZE)

        # AlphaGo Zero
        self._env = None
        self._trajectory = None
        self._lastReward = 0
        self._mcts = _Mcts(self._evalModelForMcts, trainConfig.mcts)
        self._model, self._optimizer = self._buildModel(trainConfig,
                                                        simulationConfig.gridWidth,
                                                        simulationConfig.gridHeight)
        self._lossP = CrossEntropyLoss()
        self._lossV = MSELoss()
        self._trainLoss = np.zeros((1), dtype=np.float32)

        if False:
            summary(self._model,
                    (3, simulationConfig.gridHeight + 2, simulationConfig.gridWidth + 2),
                    None)
            exit(-1)

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value
        self._mcts.initEnv(value)

    def getAction(self, observations, info):
        targetPolicy, intAction = self._mcts.search(observations, info)

        sample = (self._stateProcessing(observations),
                  intAction,
                  targetPolicy) # c'est pas bon; quand on EAT, la position de la pomme dans application est != pomme dans
                                # mcts simule!
        self._trajectory.append(sample)

        return self._gameActions[intAction]

    def train(self, observations, info, action, newObservations, newInfo, reward, done):
        self._lastReward = reward

    def onEpisodeBegin(self, episode, frameStats):
        self._trajectory = []

    def onEpisodeDone(self, episode, frameStats):
        for s in self._trajectory:
            self._replayBuffer.append((*s, self._lastReward))

        if len(self._replayBuffer) > AgentAlphaGoZero.BATCH_SIZE * 3:
            self._trainLoss = np.zeros((1), dtype=np.float32)
            self._trainFromReplayBuffer()
            self._mcts.reset()

        frameStats.loc[0, "TrainLossMin"] = self._trainLoss.min()
        frameStats.loc[0, "TrainLossMax"] = self._trainLoss.max()
        frameStats.loc[0, "TrainLossMean"] = self._trainLoss.mean()

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
        numSamples = len(self._replayBuffer)
        sampleIndices = np.arange(numSamples)

        def batchify(iterable):
            it = iter(iterable)
            yield from iter(lambda: list(islice(it, AgentAlphaGoZero.BATCH_SIZE)), [])

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

        # epochs
        for _ in range(5):

            # all samples
            np.random.shuffle(sampleIndices)
            for batchIndices in batchify(sampleIndices):
                samples = self._replayBuffer.getitems(batchIndices)
                states, _, targetPolicies, targetValues = samples

                targetPolicies = np.vstack(targetPolicies)
                targetValues = np.array(targetValues, dtype=np.float32)

                loss = self._trainBatch(unpack(states),
                                        from_numpy(targetPolicies),
                                        from_numpy(targetValues).view(-1, 1))
                self._trainLoss = np.append(self._trainLoss, loss)

    def _trainBatch(self, states, targetPolicies, targetValues):
        self._model.train()

        # gather fait un lookup, donc enleve les dimensions
        p, v = self._model(*states)

        lossP = self._lossP(p, targetPolicies)
        lossV = self._lossV(v, targetValues)
        loss = lossV + lossP

        # gradient descent
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._model.eval()

        return loss.detach().numpy()

    def _buildModel(self, trainConfig, width, height):
        numInputs = 0
        numChannels = 3

        model = _AlphaGoZeroConvNet(width, height, numChannels, numInputs, self._numGameActions)

        model.eval()
        optimizer = Adam(model.parameters(), lr=trainConfig.lr)

        return model, optimizer

    def _evalModelForMcts(self, state):
        with no_grad():
            modelArgs = self._stateProcessing(state)
            outputs = self._model(*modelArgs)

        return [o.numpy() for o in outputs]

    def _stateProcessing(self, state):
        stateProcessed = self._stateProcessor(state)
        return self._stateToTorch(*stateProcessed)

    def _stateToTorch(self, x0, x1):
        x0 = from_numpy(x0.astype(np.float32))
        x0 = unsqueeze(x0, 0)

        if not x1 is None:
            x1 = from_numpy(x1.astype(np.float32))
            x1 = unsqueeze(x1, 0)

        return x0, x1
