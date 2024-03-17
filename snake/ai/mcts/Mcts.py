import numpy as np

from snake.core import Profile
from snake.game import GameAction
from snake.configs import Rewards
from .NodeFactory import _NodeFactory


class _Mcts(object):
    def __init__(self, modelEvalCallable, trainConfig):
        self._modelEvalCallable = modelEvalCallable
        self._nodeFactory = _NodeFactory()

        self._actions = list(GameAction)
        self._numActions = len(GameAction)
        self._maxVisitCount = trainConfig.maxVisitCount

        self._cpuct = trainConfig.mcts.cpuct
        self._numSimulations = trainConfig.mcts.numSimulations
        self._temperature = trainConfig.mcts.temperature

        self.reset()

    def initEnv(self, env):
        self._env = env

    def reset(self):
        self._nodeFactory.clear()

        self.getOrCreateTotal = 0
        self.selectTotal = 0
        self.backPropagationTotal = 0
        self.expandTotal = 0
        self.modelEvalTotal = 0
        self.simResetTotal = 0
        self.simApplyTotal = 0

        # self._temperature *= 0.995
        # self._temperature = max(self._temperature, 1)

    def search(self, state, info):
        # done et won peuvent etre inconsistent avec state/info; a valider?
        root = self._nodeFactory.getOrCreate(state, info, False, False)
        root.validate(state, info)

        self.getOrCreateTotal += self._nodeFactory.getOrCreateDuration

        simulation = 0
        while simulation < self._numSimulations:
            with Profile() as selP:
                node, truncated, trajectory = self._select(root)
            self.selectTotal += selP.duration

            if truncated:
                v = 0
                simulation += 1
            elif node.done:
                v = 1 if node.won else -1
                simulation += 1
            else:
                with Profile() as expP:
                    v = self._expand(node)
                self.expandTotal += expP.duration

            with Profile() as backP:
                self._backpropagation(trajectory, v)
            self.backPropagationTotal += backP.duration

            self._nodeFactory.validateVisitCount()

        return self._choice(root)

    def _select(self, node):
        trajectory = []

        while True:
            # sanity check
            assert not node is None

            # arreter sur etat terminal, node non explore ou cycle detecte
            truncated = node.vistCount >= self._maxVisitCount
            if node.done or node.isLeaf or truncated:
                break

            node.vistCount += 1

            # selection action selon upper confidence bound
            ucb = self._ucb(node)

            # greedy selection d'action
            # mais limiter aux choix valides
            intAction = _Mcts._limitArgmax(ucb, node)

            # ajouter choix a la trajectoire
            trajectory.append((node, intAction))

            # passer a la node suivante
            node = node.child[intAction]

        return node, truncated, trajectory

    def _expand(self, node):
        # initialize les informations pour les 'edges'
        node.Q = np.zeros(self._numActions, dtype=np.float32)
        node.N = np.zeros(self._numActions, dtype=np.float32)
        node.W = np.zeros(self._numActions, dtype=np.float32)


        with Profile() as modelP:
            # evaluer P et V
            p, v = self._modelEvalCallable(node.state)
        self.modelEvalTotal += modelP.duration

        p = p.squeeze()
        v = v.item()

        # s'assurer que P n'a que les actions permises
        actionsAvailable = node.state["available_actions"]
        node.P = np.zeros_like(actionsAvailable, dtype=np.float32)

        # s'assurer que v est [-1, 1]; papier origine veut -1 == partie perdue et 1 == partie gagnee
        # important que ce soit ce range pour que la partie de _ucb() traitant du # de visites des nodes
        # ait le meme poids relatif que Q(s, a)
        v = max(min(v, 1), -1)

        # construire les nodes pour chaque actions debutant a node.state
        node.child = []
        for i, (action, available) in enumerate(zip(self._actions, actionsAvailable)):
            if available == 0:
                # action non permise, mettre None dans cette branche
                newNode = None
            else:
                with Profile() as resetP:
                    # trouver l'etat correspondand a l'action desiree
                    self._env.reset(options=node.state)

                self.simResetTotal += resetP.duration

                with Profile() as stepP:
                    newObservations, _, terminated, truncated, newInfos = self._env.step(action)
                    done = terminated or truncated
                self.simApplyTotal += stepP.duration

                # si l'etat a deja ete visite, le partager
                # sinon en creer un nouveau
                won = newInfos["reward_type"] == Rewards.WIN
                newNode = self._nodeFactory.getOrCreate(newObservations, newInfos, done, won)

                self.getOrCreateTotal += self._nodeFactory.getOrCreateDuration

                node.P[i] = p[i]

            node.child.append(newNode)

        node.P = node.P / node.P.sum()

        return v

    def _backpropagation(self, trajectory, v):
        for node, action in reversed(trajectory):
            # marquer node comme non visite
            node.vistCount = 0

            # ajouter une visite
            node.N[action] += 1
            node.W[action] += v

            # mettre a jour fonction valeur action
            node.Q[action] = node.W[action] / node.N[action]

    def _choice(self, node):
        # obtenir policy amelioree
        exponent = 1.0 / self._temperature
        n = node.N ** exponent
        newPolicy = n / n.sum()

        # limiter aux actions valides de node
        availableActions = node.state["available_actions"]
        newPolicy = newPolicy * availableActions
        newPolicy = newPolicy / newPolicy.sum()

        # selection random
        intAction = np.random.choice(self._numActions, p=newPolicy)

        return newPolicy, intAction

    def _ucb(self, node):
        return node.Q + (self._cpuct * node.P * np.sqrt(node.N.sum()) / (1 + node.N))

    @staticmethod
    def _limitArgmax(values, node):
        # ne garder que les actions possibles (!= 0)
        availableActions = node.state["available_actions"]
        availableActions = np.nonzero(availableActions)[0]

        possibleActions = values[availableActions]
        intAction = np.argmax(possibleActions)

        # remapper action vers le bon index
        intAction = availableActions[intAction]
        return intAction
