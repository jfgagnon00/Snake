"""
Gere la serialisation des actions pour l'enregistrement/playback
"""

from json import JSONDecoder, JSONEncoder
from snake.game import GameAction


class _TimedAction(object):
    def __init__(self, time, action):
        self.time = time
        self.action = action

    def __str__(self):
        return f"{self.time}, {self.action.name}"

class _TimedActionEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, _TimedAction):
            return {"time": o.time, "action": o.action.name}
        return super().default(o)

class _TimedActionDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, object_hook=self._object_hook)

    def _object_hook(self, dict_):
        if len(dict_.keys()) == 2 and "time" in dict_ and "action" in dict_:
            return _TimedAction(dict_["time"], GameAction[dict_["action"]])

        return dict_