from numpy.random import randint
from snake.game.GameSimulation import ResetException

# TODO: semble plus approprite dans application
# n'a pas vraiment de lien avec l'agent: ca genere des nouveaux samples
# et a une dependance sur la simulation
class _HindsightExperienceReplay():
    """
    Implementation de Hindsight Experience Replay.
    """

    def __init__(self, futureK):
        self._futureK = futureK
        self._env = None

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    def append(self, state, info, action, newState, newInfo, done):
        if not self._env is None:
            self._episode.append((state, info, action, newState, newInfo, done))

    def clear(self):
        self._episode = []

    def replay(self):
        if True:
            be = len(self._episode)
            bi = be // 2 + 1

            for t in range(bi, be):
                startState, startInfo, action, endState, _, episodeDone = self._episode[t]
                if episodeDone:
                    continue

                try:
                    observations, _ = self._env.reset(options={
                        "score": startState["score"],
                        "food_position": endState["head_position"],
                        "head_direction": startState["head_direction"],
                        "snake_bodyparts": startInfo["snake_bodyparts"],
                    })
                except ResetException as e:
                    continue

                newObservations, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                self._env.render()

                yield observations, action, newObservations, reward, done
        else:
            for s in range(len(self._episode)):
                goalState, goalInfo, action, _, _, goalDone = self._episode[s]
                if goalDone:
                    continue

                transitions = self._sampleTransitions(s + 1)
                for t in transitions:
                    _, _, _, newGoalState, _, newGoalDone = self._episode[t]
                    if newGoalDone:
                        continue

                    try:
                        observations, _ = self._env.reset(options={
                            "score": goalState["score"],
                            "food_position": newGoalState["head_position"],
                            "head_direction": goalState["head_direction"],
                            "snake_bodyparts": goalInfo["snake_bodyparts"],
                        })
                    except ResetException:
                        # certaines combinaisons ne sont pas valide
                        # reset() lance une exception; simplement ignorer
                        continue

                    newObservations, reward, terminated, truncated, _ = self._env.step(action)
                    done = terminated or truncated

                    self._env.render()

                    yield observations, action, newObservations, reward, done

    def _sampleTransitions(self, start):
        stop = len(self._episode)
        count = min(self._futureK, stop - start)
        if count > 0:
            return randint(start, stop, size=count)

        return []
