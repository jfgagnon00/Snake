import numpy as np
import os
import time

from itertools import islice
from torch import from_numpy, \
                no_grad, \
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
from snake.configs import Rewards
from snake.ai.agents.AgentBase import AgentBase
from snake.ai.mcts import _Mcts
from snake.ai.nets import _AlphaGoZeroConvNet
from snake.ai.ReplayBuffer import _ReplayBuffer
from snake.ai.StateProcessor import _StateProcessor


class AgentAlphaGoZero(AgentBase):
    MEMORY_SIZE = 8192
    BATCH_SIZE = 64
    BATCH_COUNT = 30
    TRAIN_EPOCH = 8

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
        self._trajectory = None
        self._lastReward = 0
        self._mcts = _Mcts(self._evalModelForMcts, trainConfig)
        self._model, self._optimizer = self._buildModel(trainConfig,
                                                        simulationConfig.gridWidth,
                                                        simulationConfig.gridHeight)
        self._lossP = CrossEntropyLoss(reduction="none")
        self._lossV = MSELoss(reduction="none")
        self._trainLoss = np.zeros((1), dtype=np.float32)
        self._numTrain = 0

        if False:
            summary(self._model,
                    (3, simulationConfig.gridHeight + 2, simulationConfig.gridWidth + 2),
                    None)
            exit(-1)

    def getAction(self, observations, infos):
        targetPolicy, intAction = self._mcts.search(observations, infos)

        sample = (self._stateProcessing(observations, infos),
                  intAction,
                  targetPolicy) # c'est pas bon; quand on EAT, la position de la pomme dans application est != pomme dans
                               # mcts simule!
        self._trajectory.append(sample)

        return self._gameActions[intAction]

    def train(self, observations, info, action, newObservations, newInfo, reward, done):
        if newInfo["reward_type"] == Rewards.WIN:
            self._lastReward = 1
        elif newInfo["reward_type"] == Rewards.TRUNCATED:
            self._lastReward = 0
        else:
            self._lastReward = -1

    def onEpisodeBegin(self, episode, frameStats):
        self._trajectory = []

    def onEpisodeDone(self, episode, frameStats):
        for s in self._trajectory:
            self._replayBuffer.append((*s, self._lastReward))

        count = AgentAlphaGoZero.BATCH_COUNT * AgentAlphaGoZero.BATCH_SIZE
        if len(self._replayBuffer) > count:
            print("mcts stats")
            print("    getOrCreateTotal:", self._mcts.getOrCreateTotal)
            print("    selectTotal:", self._mcts.selectTotal)
            print("    backPropagationTotal:", self._mcts.backPropagationTotal)
            print("    expandTotal:", self._mcts.expandTotal)
            print("    expandCount:", self._mcts.expandCount)
            print("    expand:", self._mcts.expandTotal / self._mcts.expandCount)
            print("    numTrain:", self._numTrain)
            print("    len replay:", len(self._replayBuffer))
            print("    num nodes", len(self._mcts._nodeFactory._stateToNode))
            print("    modelEvalTotal:", self._mcts.modelEvalTotal)
            print("    simApplyTotal:", self._mcts.simApplyTotal)
            print()

            self._numTrain += 1
            self._trainLoss = np.zeros((1), dtype=np.float32)
            self._trainFromReplayBuffer()
            self._mcts.reset()
            self._replayBuffer.clear()


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

        for _ in range(AgentAlphaGoZero.TRAIN_EPOCH):
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
        loss = (lossV + lossP).mean()

        # gradient descent
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._model.eval()

        return loss.detach().numpy()

    def _buildModel(self, trainConfig, width, height):
        numInputs = 0
        numChannels = 4 if trainConfig.frameStack > 1 else 3

        model = _AlphaGoZeroConvNet(width, height, numChannels, numInputs, self._numGameActions)

        model.eval()
        optimizer = Adam(model.parameters(), lr=trainConfig.lr)

        return model, optimizer

    def _evalModelForMcts(self, state):
        with no_grad():
            modelArgs = self._stateProcessing(state, state)
            outputs = self._model(*modelArgs)

        return [o.numpy() for o in outputs]

    def _stateProcessing(self, state, info):
        stateProcessed = self._stateProcessor(state, info)
        return self._stateToTorch(*stateProcessed)

    def _stateToTorch(self, x0, x1):
        x0 = from_numpy(x0.astype(np.float32))
        x0 = unsqueeze(x0, 0)

        if not x1 is None:
            x1 = from_numpy(x1.astype(np.float32))
            x1 = unsqueeze(x1, 0)

        return x0, x1
