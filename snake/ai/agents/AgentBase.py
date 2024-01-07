class AgentBase(object):
    """
    Classe de base pour tous les agents.
    """
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        """
        Appeler pour reseter les etats
        """
        pass

    def getAction(self, *args):
        """
        Appeler pour avoir une action
        """
        pass

    def onEpisodeBegin(self, *args):
        """
        Appeler quand un episode/partie demarre (tout de suite apres reset)
        """
        pass

    def onEpisodeDone(self, *args):
        """
        Appeler quand un episode/partie est termine
        """
        pass

    def train(self, *args):
        """
        Appeler pour faire l'entrainement
        """
        pass

    def save(self, *args):
        """
        Appeler pour sauvegarder l'etat courant
        """
        pass
