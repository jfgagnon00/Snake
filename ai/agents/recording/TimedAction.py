from json import JSONDecoder, JSONEncoder


class _TimedAction():
    def __init__(self, time, action):
        self.time = time
        self.action = action

    def __str__(self):
        return f"{self.time}, {self.action}"

class _TimedActionEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, _TimedAction):
            return {"time": o.time, "action": str(o.action)}
        return super().default(o)
