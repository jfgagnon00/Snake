class GameEnvironment():
    """
    Responsable d'appliquer le mouvement au serpent
    """

    MOVEMENT_TURN_LEFT = 0
    MOVEMENT_TURN_RIGHT = 1
    MOVEMENT_FORWARD = 2

    def __init__(self, gameConfig):
        self._grid = ?? # array 2d
        self._snake = ??
        self._food = ??
        self._score = ??
        self._rewards = ??

    def reset(self):
        """
        Remet l'environment dans un etat initial
        """
        pass

    def apply(self, movement):
        """
        Applique le movement au serpent et met a jour les etats internes
        """
        # 1. bouge serpent
        # 2. mettre a jour grid
        # 3. resoudre collision
        # 4. calculer reward et game over
        pass
