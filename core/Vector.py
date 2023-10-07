import numpy as np

class Vector():
    """
    Represente un vecteur/point 2D
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def to_numpy(self):
        """
        Converti en array numpy
        """
        # TODO: sous optimal, devrait heriter de np.array plutot que creer des objets
        return np.array([self.x, self.y])
