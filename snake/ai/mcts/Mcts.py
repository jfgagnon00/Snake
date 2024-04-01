import numpy as np

from snake.core import Profile, Vector
from snake.game import GameAction, GameSimulation
from .NodeFactory import _NodeFactory


class _Mcts(object):
    def __init__(self, trainConfig):
        self._nodeFactory = _NodeFactory()

        self._simulation = GameSimulation()
        self._simulation.winDelegate.register(self._onWin)
        self._simulation.loseDelegate.register(self._onLose)

        self._actions = list(GameAction)
        self._numActions = len(GameAction)

        self._cpuct = trainConfig.mcts.cpuct
        self._numExpands = trainConfig.mcts.numExpands

        self.reset()

    def reset(self):
        # self._nodeFactory.clear()

        self.getOrCreateTotal = 0
        self.selectTotal = 0
        self.backPropagationTotal = 0
        self.expandTotal = 0
        self.expandCount = 0

    def search(self, state, info):
        root = self._getOrCreate(state, info, False, False)

        for _ in range(self._numExpands):
            trajectory, truncated = self._select(root)
            assert len(trajectory) > 0

            if truncated:
                v = 0
            else:
                node, intAction = trajectory[-1]
                if intAction is None:
                    v = 1 if node.won else -1
                else:
                    v = self._expand(trajectory)

            self._backpropagation(trajectory, v)

        return self._choice(root)

    def _select(self, node):
        with Profile() as selP:
            assert not node is None

            trajectory = []
            truncated = False

            while True:
                if node == None:
                    break

                if node.visitCount >= 1: #node.simulationState.snake.length:
                    truncated = True
                    break

                if node.done:
                    trajectory.append((node, None))
                    break

                node.visitCount += 1

                # selection action selon upper confidence bound
                ucb = self._ucb(node)

                # greedy selection d'action
                # mais limiter aux choix valides
                intAction = _Mcts._limitArgmax(ucb, node)

                # ajouter choix a la trajectoire
                trajectory.append((node, intAction))

                # passer a la node suivante
                node = node.child[intAction]

        self.selectTotal += selP.duration

        return trajectory, truncated

    def _expand(self, trajectory):
        with Profile() as expP:
            assert not trajectory is None
            assert len(trajectory) > 0

            node, intAction = trajectory[-1]

            # s'assurer que la node peut etre expanded
            assert not node.done

            # s'assurer que la node enfant a expanded est valide
            assert node.child[intAction] is None
            assert node.availableActions[intAction] == 1

            playoutTrajectory, truncated = self._playout(node, intAction)
            v = 0

            # valider trajectoire
            if len(playoutTrajectory) > 0:
                node.child[intAction] = playoutTrajectory[0][0]
                lastNode, lastAction = playoutTrajectory[-1]

                if not truncated:
                    assert lastAction is None
                    assert lastNode.done

                    v = 1 if lastNode.won else -1

                # warmup des nodes pour le prochain playout
                self._backpropagation(playoutTrajectory, v)

                # tirer profit de la derniere action
                # early out potentiel pour le prochain playout
                self._backpropagateAvailableActions(playoutTrajectory)

        self.expandTotal += expP.duration
        self.expandCount += 1

        return v

    def _playout(self, node, intAction):
        trajectory = []
        truncated = False
        self._done = False
        self._won = False

        while True:
            newObservations, newInfos = self._applyOneAction(node, intAction)
            node = self._getOrCreate(newObservations, newInfos, self._done, self._won)

            if node.visitCount > 0:
                truncated = True
                break

            if node.done:
                trajectory.append( (node, None) )
                break

            node.visitCount += 1
            intAction = self._getIntAction(node)
            trajectory.append( (node, intAction) )

        return trajectory, truncated

    def _getOrCreate(self, state, info, done, won):
        with Profile() as getOrCreateP:
            newNode = self._nodeFactory.getOrCreate(state, info, done, won)
            newNode.validate(state, info)
        self.getOrCreateTotal += getOrCreateP.duration
        return newNode

    def _applyOneAction(self, node, intAction):
        assert node.availableActions[intAction] == 1

        action = self._actions[intAction]
        newState = self._simulation.apply(action, node.simulationState, inplace=False)
        newObservations = self._simulation.getObservations(newState)
        newInfos = self._simulation.getInfos(newState, copy=False)

        newActionsAvailable = newState.availableActions()
        self._done = self._done or newActionsAvailable.sum() == 0
        newInfos["available_actions"] = newActionsAvailable

        return newObservations, newInfos

    def _getIntAction(self, node):
        head = node.simulationState.snake.head
        direction = node.simulationState.snake.direction
        food = node.simulationState.food
        head2food = food - head

        # prendre action qui approche de la destination
        # si elle est possible, sinon prendre au hasard
        k = Vector.krot90(direction, head2food)
        action = GameAction.fromKRot90(k)
        intAction = self._actions.index(action)

        if node.availableActions[intAction] == 1 and \
           not (action == GameAction.FORWARD and Vector.dot(direction, head2food) < 0):
            return intAction

        availableActions = node.availableActions
        p = availableActions / availableActions.sum()
        intAction = np.random.choice(self._numActions, p=p)

        assert availableActions[intAction] == 1

        return intAction

    def _backpropagateAvailableActions(self, trajectory):
        lastNode = None

        for node, intAction in reversed(trajectory):
            if lastNode is None:
                if node.won:
                    break
                lastNode = node
                continue

            if lastNode.done:
                node.availableActions[intAction] = 0
                node.done = node.availableActions.sum() == 0
                lastNode = node
            else:
                break

    def _backpropagation(self, trajectory, v):
        with Profile() as backP:
            for node, intAction in reversed(trajectory):
                node.visitCount = 0

                if not intAction is None:
                    # ajouter une visite
                    node.N[intAction] += 1
                    node.W[intAction] += v

                    # mettre a jour fonction valeur action
                    node.Q[intAction] = node.W[intAction] / node.N[intAction]

        self.backPropagationTotal += backP.duration

    def _choice(self, node):
        # obtenir policy amelioree
        # limiter aux actions valides de node
        newPolicy = node.N * node.availableActions
        if node.done:
            return newPolicy, 2

        newPolicy = newPolicy / newPolicy.sum()

        # selection random
        intAction = np.random.choice(self._numActions, p=newPolicy)

        return newPolicy, intAction

    def _ucb(self, node):
        return node.Q + (self._cpuct * node.availableActions * np.sqrt(node.N.sum()) / (1 + node.N))

    @staticmethod
    def _limitArgmax(values, node):
        # ne garder que les actions possibles (!= 0)
        availableActions = node.availableActions
        availableActions = np.nonzero(availableActions)[0]

        possibleActions = values[availableActions]
        intAction = np.argmax(possibleActions)

        # remapper action vers le bon index
        intAction = availableActions[intAction]
        return intAction.item()

    def _onWin(self):
        self._done = True
        self._won = True

    def _onLose(self):
        self._done = True
        self._won = False
