from snake.core import Delegate
from .GameAction import GameAction
from .GameSimulationState import GameSimulationState


class GameSimulation(object):
    """
    Responsable de la logique de simulation du jeu de serpent.
    La simulation evolue sur une grille discrete.
    """
    def __init__(self):
        self._outOfBoundsDelegate = Delegate()
        self._collisionDelegate = Delegate()
        self._eatDelegate = Delegate()
        self._winDelegate = Delegate()
        self._loseDelegate = Delegate()
        self._turnDelegate = Delegate()
        self._moveDelegate = Delegate()

    @property
    def outOfBoundsDelegate(self):
        return self._outOfBoundsDelegate

    @property
    def collisionDelegate(self):
        return self._collisionDelegate

    @property
    def eatDelegate(self):
        return self._eatDelegate

    @property
    def winDelegate(self):
        return self._winDelegate

    @property
    def loseDelegate(self):
        return self._loseDelegate

    @property
    def turnDelegate(self):
        return self._turnDelegate

    @property
    def moveDelegate(self):
        return self._moveDelegate

    def apply(self, action, state, inplace=True):
        """
        Met a jour la simulation en fonction de l'action fournie. Les consequences de
        l'action appliquee sont communique par les divers delegate.

        Params
            action: GameAction a appliquee
            state: GameSimulationState a partir duquel action est appliquee
            inplace: Si True, applique action sur state. Si False, retourne
                     une copie de state avec action applique.
        """
        if not inplace:
            state = state.copy()

        direction = state.snake.direction

        if action != GameAction.FORWARD:
            k = action.krot90
            direction = direction.rot90(k)
            self._turnDelegate()

        # bouger la tete dans la nouvelle direction
        # ATTENTION: l'operateur + cree une nouvelle instance
        head = state.snake.head + direction

        if head == state.food:
            # tete est sur la nourriture, grandire le serpent
            state.snake.bodyParts.appendleft(head)

            cellCount = state.gridWidth * state.gridHeight
            if state.snake.length == cellCount:
                state.food = None
                # serpent couvre toutes les cellules, gagner la partie
                self._winDelegate()
            else:
                state.food = GameSimulationState.randomFood(state)
                self._eatDelegate()

            return state

        if state.outOfBound(head):
           # tete est en dehors de la grille, terminer
           self._outOfBoundsDelegate()
           self._loseDelegate()
           return state

        if state.collisonTest(head):
            # tete est en collision
            self._collisionDelegate()
            self._loseDelegate()
            return state

        # bouger le corps du serpent
        state.snake.bodyParts.pop()
        state.snake.bodyParts.appendleft(head)
        self._moveDelegate()

        return state

    def getObservations(self, state):
        return {
            "dummy": 0
        }

    def getInfos(self, state, copy=True):
        return {
            "simulation_state": state.copy() if copy else state,
        }
