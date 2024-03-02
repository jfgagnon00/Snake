from io import StringIO
from pprint import pprint


class _MctsNodeException(Exception):
    def __init__(self, message, state=None, info=None, nodeState=None):
        super().__init__(message)
        self._state = state
        self._info = info
        self._nodeState = nodeState

    def __str__(self):
        with StringIO() as stream:
            print(file=stream)
            print(super().__str__(), file=stream)

            if self._state is None:
                print("No state", file=stream)
            else:
                print("State", file=stream)
                pprint(self._state, stream=stream)

            if self._info is None:
                print("No info", file=stream)
            else:
                print("Info", file=stream)
                pprint(self._info, stream=stream)

            if self._nodeState is None:
                print("No node state", file=stream)
            else:
                print("Node State", file=stream)
                pprint(self._nodeState, stream=stream)

            return stream.getvalue()
