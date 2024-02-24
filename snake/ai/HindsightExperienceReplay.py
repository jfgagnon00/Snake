from pprint import pprint

from copy import deepcopy
from gymnasium import make as gym_Make
from numpy.random import randint


class _HindsightExperienceReplay():
    """
    Implementation de Hindsight Experience Replay.
    """

    def __init__(self, configs):
        self._futureK = configs.train.hindsightFutureK

        # l'agent devrait supporter l'apprentissage par 'goal'. Dans le
        # present contexte, il est plus simple de resimuler une action
        # avec des modifications sur les observations.
        self._env = gym_Make("snake/SnakeEnvironment-v0",
                             renderMode=None,
                             environmentConfig=configs.environment,
                             simulationConfig=configs.simulation,
                             graphicsConfig=configs.graphics,
                             trainConfig=configs.train)

    def append(self, state, info, action, done):
        self._episode.append((state, info, action, done))

    def clear(self):
        self._episode = []

    def replay(self):
        for s in range(len(self._episode)):
            goalState, goalInfo, action, goalDone = self._episode[s]
            if goalDone:
                continue

            transitions = self._sampleTransitions(s + 1)
            for t in transitions:
                newGoalState, _, _, newGoalDone = self._episode[t]
                if newGoalDone:
                    continue

                observations, _ = self._env.reset(options={
                    "score": goalState["score"],
                    "food_position": newGoalState["head_position"],
                    "head_direction": goalState["head_direction"],
                    "snake_bodyparts": goalInfo["snake_bodyparts"],
                })

                newObservations, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                yield observations, action, newObservations, reward, done

    def _sampleTransitions(self, start):
        if self._futureK == 1:
            return [-1]

        stop = len(self._episode)
        count = min(self._futureK, stop - start)
        if count > 0:
            return randint(start, stop, size=count)

        return []
