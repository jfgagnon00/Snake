import json
import os
import random

from ai.agents import AgentBase
from datetime import datetime
from .TimedAction import _TimedAction, _TimedActionEncoder


class AgentActionRecorder(AgentBase):
    """
    Agent qui encapsule un autre agent dans le but de
    capturer ses actions pour les enregistrer dans un fichier.
    Ne pas utiliser comme agent conventionel.
    """
    def __init__(self, agent, recordPattern, recordN=None):
        super().__init__()
        self._episode = 0
        self._agent = agent
        self._recordPattern = recordPattern
        self._episodeCountModulo = 1 if recordN is None else int(recordN)
        self.reset()

    @property
    def saveDelegate(self):
        return self._saveDelegate

    def reset(self):
        """
        Redemarre la capture
        """
        self._resetInternal()
        random.seed(self._seed)

        self._agent.reset()

    def getAction(self, *args):
        """
        Enregistre l'action de l'agent encapsule
        """
        action = self._agent.getAction(*args)

        self._time += 1
        if self._isEmpty() or self._timedActions[-1].action != action:
            self._timedActions.append(_TimedAction(self._time, action))

        return action

    def onEpisodeDone(self, episode):
        self._agent.onEpisodeDone(episode)

        self._episode = episode
        if not self._isEmpty() and (episode % self._episodeCountModulo) == 0:
            self.save()

    def save(self):
        filename = self._recordPattern.replace("%", str(self._episode))

        self._agent.save(filename)

        path, _ = os.path.split(filename)
        if not path is None:
            os.makedirs(path, exist_ok=True)

        with open(filename, "w") as file:
            json.dump({"seed": self._seed,
                        "timedActions": self._timedActions},
                file,
                cls=_TimedActionEncoder,
                indent=4)

        self._resetInternal()

    def _isEmpty(self):
        return len(self._timedActions) == 0

    def _resetInternal(self):
        self._time = -1
        self._seed = datetime.now().timestamp()
        self._timedActions = []
