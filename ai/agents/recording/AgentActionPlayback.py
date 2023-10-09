import json
import os
import random

from ..AgentBase import AgentBase
from .TimedAction import _TimedActionDecoder


class AgentActionPlayback(AgentBase):
    """
    Agent qui rejoue les actions enregistres par AgentActionRecorder
    """
    def __init__(self, filename):
        with open(filename, "r") as file:
            dict_ = json.load(file, cls=_TimedActionDecoder)
            self._seed = dict_["seed"]
            self._timedActions = dict_["timedActions"]
        self.reset()

    def reset(self):
        """
        Redemarre la capture
        """
        self._time = -1
        self._nextIndex = 0
        self._next()
        random.seed(self._seed)

    def getAction(self, *args):
        """
        Relie les actions enregistrees
        """
        self._time += 1

        if self._nextTime == self._time:
            self._next()

        return self._currentAction

    def _next(self):
        self._currentAction = self._timedActions[self._nextIndex].action

        nextIndex = self._nextIndex + 1
        if nextIndex < len(self._timedActions):
            self._nextTime = self._timedActions[nextIndex].time
            self._nextIndex = nextIndex
