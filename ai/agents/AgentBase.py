class AgentBase():
    """
    Classe de base pour tous les agents.
    """
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        """
        Appeler pour reseter les etats (ou quant une simulation va demarrer)
        """
        pass

    def train(self, *args):
        pass

    def getAction(self, *args):
        """
        Appeler pour avoir une action
        """
        pass

    def onSimulationDone(self):
        """
        Appeler quand la simulation est terminee mais avant reset
        """
        pass
