import json
import os

from snake.ai.agents import AgentBase
from snake.core import RandomProxy
from .Random import _RandomRecorder
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
        self._randomRecorder = RandomProxy.instance = _RandomRecorder(RandomProxy.instance)
        self.reset()

    def reset(self):
        """
        Redemarre la capture
        """
        self._time = -1
        self._timedActions = []
        self._randomRecorder.reset()
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

    def onEpisodeBegin(self, *args):
        self._agent.onEpisodeBegin(*args)

    def onEpisodeDone(self, episode, *args):
        self._agent.onEpisodeDone(episode, *args)

        self._episode = episode
        if not self._isEmpty() and (episode % self._episodeCountModulo) == 0:
            self.save()

    def train(self, *args):
        self._agent.train(*args)

    def save(self):
        filename = self._recordPattern.replace("%", str(self._episode))

        self._agent.save(filename)

        path, _ = os.path.split(filename)
        if not path is None:
            os.makedirs(path, exist_ok=True)

        with open(filename, "w") as file:
            json.dump({"timedActions": self._timedActions,
                       "random_choices": self._randomRecorder.choices},
                       file,
                       cls=_TimedActionEncoder,
                       indent=4)

    def _isEmpty(self):
        return len(self._timedActions) == 0
