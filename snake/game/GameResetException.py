from io import StringIO
from pprint import pprint

class GameResetException(Exception):
    def __init__(self, message, occupancyGrid, position):
        super().__init__(message)
        self.occupancyGrid = occupancyGrid
        self.position = position

    def __str__(self):
        with StringIO() as stream:
            print(file=stream)
            print(super().__str__(), file=stream)
            if self.occupancyGrid is None:
                print("No occupancy grid", file=stream)
            else:
                print("occupancyGrid shape", self.occupancyGrid.shape, file=stream)
                print("occupancyGrid", file=stream)
                pprint(self.occupancyGrid, stream=stream)
            print("Position", self.position, file=stream)
            return stream.getvalue()
