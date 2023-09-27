from . import GameEnvironment

class InteractiveApplication():
    """
    Responsable du UI et flow pour GameEnvironment
    """
    def __init__(self, agent):
        self._environment = GameEnvironment()
        self._agent = agent
        self._done = False
        # fenetre pygame

    def run(self):
        while not self._done:
            self.update()

    def update(self):
        # 1. lire input (clavier et autres)
        
        # 2. demander a l'agent un movement
        action = self._agent.get_next_move(self._environment.get_state)
        
        # 3. appliquer movement sur environment
        self._environment.apply(action)

        # 3.5 enregistrer le playback
        
        # 4. refresh display
        pass