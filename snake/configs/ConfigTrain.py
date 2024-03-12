from .ConfigMcts import ConfigMcts


class ConfigTrain(object):
    def __init__(self):
        self.gamma = 0.0
        self.epsilon = 0.0
        self.epsilonDecay = 0.0
        self.epsilonMin = 0.0
        self.tau = 0
        self.lr = 0.0
        self.agent = "AgentRandom"
        self.unattended = False
        self.episodes = 1
        self.maxVisitCount = 0
        self.frameStack = 0
        self.showStats = True
        self.mcts = ConfigMcts()
