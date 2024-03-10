import numpy as np

from snake.game import GameAction
from snake.configs import Rewards
from .NodeFactory import _NodeFactory


class _Mcts(object):
    def __init__(self, modelEvalCallable, mctsConfig):
        self._modelEvalCallable = modelEvalCallable
        self._nodeFactory = _NodeFactory()

        self._actions = list(GameAction)
        self._numActions = len(GameAction)

        self._cpuct = mctsConfig.cpuct
        self._numSimulations = mctsConfig.numSimulations
        self._temperature = mctsConfig.temperature

    def initEnv(self, env):
        self._env = env

    def reset(self):
        self._nodeFactory.clear()

    def search(self, state, info):
        # done et won peuvent etre inconsistent avec state/info; a valider?
        root = self._nodeFactory.getOrCreate(state, info, False, False)
        root.validate(state, info)

        simulation = 0
        while simulation < self._numSimulations:
            node, truncated, trajectory = self._select(root)

            if truncated:
                v = 0
                simulation += 1
            elif node.done:
                v = 1 if node.won else -1
                simulation += 1
            else:
                v = self._expand(node)

            self._backpropagation(trajectory, v)
            self._nodeFactory.validateVisitCount()

        return self._choice(root)

    def _select(self, node):
        trajectory = []

        while True:
            # sanity check
            assert not node is None

            # arreter sur etat terminal, node non explore ou cycle detecte
            truncated = node.vistCount >= 1
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

        # evaluer P et V
        p, v = self._modelEvalCallable(node.state)
        v = v.item()

        # s'assurer que P n'a que les actions permises
        availableActions = node.state["available_actions"]
        node.P = p.squeeze() * availableActions
        node.P = node.P / node.P.sum()

        # s'assurer que v est [-1, 1]; papier origine veut -1 == partie perdue et 1 == partie gagnee
        # important que ce soit ce range pour que la partie de _ucb() traitant du # de visites des nodes
        # ait le meme poids relatif que Q(s, a)
        v = max(min(v, 1), -1)

        # construire les nodes pour chaque actions debutant a node.state
        node.child = []
        for action, available in zip(self._actions, availableActions):
            if available == 0:
                # action non permise, mettre None dans cette branche
                newNode = None
            else:
                # trouver l'etat correspondand a l'action desiree
                self._env.reset(options=node.state)
                newObservations, _, terminated, truncated, newInfos = self._env.step(action)
                done = terminated or truncated

                # si l'etat a deja ete visite, le partager
                # sinon en creer un nouveau
                won = newInfos["reward_type"] == Rewards.WIN
                newNode = self._nodeFactory.getOrCreate(newObservations, newInfos, done, won)

            node.child.append(newNode)

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
        return node.Q + self._cpuct * node.P * np.sqrt(node.N.sum()) / (1 + node.N)

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
