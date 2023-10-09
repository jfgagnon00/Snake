import json
import os
import random

from datetime import datetime
from .TimedAction import _TimedAction, _TimedActionEncoder


class AgentActionRecorder():
    """
    Agent qui encapsule un autre agent dans le but de
    capturer ses actions pour les enregistrer dans un fichier
    """
    def __init__(self, agent):
        self._agent = agent
        self.reset()

    @property
    def isEmpty(self):
        return len(self._timedActions) == 0

    def reset(self):
        """
        Redemarre la capture
        """
        self._time = -1
        self._seed = datetime.now().timestamp()
        self._timedActions = []
        random.seed(self._seed)

    def getAction(self, *args):
        action = self._agent.getAction(*args)

        self._time += 1
        if self.isEmpty or self._timedActions[-1].action != action:
            self._timedActions.append(_TimedAction(self._time, action))

        return action

    def serialize(self, filename):
        path, name = os.path.split(filename)
        if not path is None:
            os.makedirs(path, exist_ok=True)

        with open(filename, "w") as file:
            json.dump({
                "seed": self._seed,
                "timedActions": self._timedActions},
                file,
                cls=_TimedActionEncoder,
                indent=4)
