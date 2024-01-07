import json

from snake.ai.agents import AgentBase
from snake.core import RandomProxy
from .Random import _RandomPlayback
from .TimedAction import _TimedActionDecoder


class AgentActionPlayback(AgentBase):
    """
    Agent qui rejoue les actions enregistres par AgentActionRecorder.
    Ne pas utiliser comme agent conventionel.
    """
    def __init__(self, filename):
        super().__init__()
        with open(filename, "r") as file:
            dict_ = json.load(file, cls=_TimedActionDecoder)
            self._timedActions = dict_["timedActions"]
            random_choices = dict_["random_choices"]

        self._randomPlayback = RandomProxy.instance = _RandomPlayback(random_choices)

        self.reset()

    def reset(self):
        """
        Redemarre la capture
        """
        self._time = -1
        self._nextTime = -1
        self._nextIndex = 0
        self._next()
        self._randomPlayback.reset()

    def getAction(self, *args):
        """
        Rejoue les actions enregistrees
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
