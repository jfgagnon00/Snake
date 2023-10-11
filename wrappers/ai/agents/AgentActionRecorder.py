import json
import os
import random

from datetime import datetime
from ai.agents import AgentBase
from .TimedAction import _TimedAction, _TimedActionEncoder


class AgentActionRecorder(AgentBase):
    """
    Agent qui encapsule un autre agent dans le but de
    capturer ses actions pour les enregistrer dans un fichier
    """
    def __init__(self, agent, recordPattern):
        self._agent = agent
        self._simulationCount = -1
        self.recordPattern = recordPattern
        self.reset()

    def reset(self):
        """
        Redemarre la capture
        """
        self._time = -1
        self._seed = datetime.now().timestamp()
        self._timedActions = []
        random.seed(self._seed)

    def getAction(self, *args):
        """
        Enregistre l'action de l'agent encapsule
        """
        action = self._agent.getAction(*args)

        self._time += 1
        if self._isEmpty() or self._timedActions[-1].action != action:
            self._timedActions.append(_TimedAction(self._time, action))

        return action

    def onSimulationDone(self):
        """
        House keeping
        """
        self._simulationCount += 1

        if not self._isEmpty():
            filename = self.recordPattern.replace("%", f"{self._simulationCount:04d}")

            path, _ = os.path.split(filename)
            if not path is None:
                os.makedirs(path, exist_ok=True)

            with open(filename, "w") as file:
                json.dump({"seed": self._seed,
                          "timedActions": self._timedActions},
                    file,
                    cls=_TimedActionEncoder,
                    indent=4)

    def _isEmpty(self):
        return len(self._timedActions) == 0
