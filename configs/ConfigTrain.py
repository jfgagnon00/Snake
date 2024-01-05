class ConfigTrain():
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.betaAnnealingSteps = 50
        self.gamma = 0.0
        self.epsilon = 0.0
        self.epsilonDecay = 0.0
        self.lr = 0.0
        self.agent = "AgentRandom"
        self.unattended = False
        self.episodes = 1
        self.episodeMaxLen = 0
        self.showStats = True
        self.useConv = False
