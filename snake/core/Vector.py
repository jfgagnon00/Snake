import numpy as np

from math import sqrt

class Vector(object):
    """
    Represente un vecteur/point 2D
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Vector(self.x, self.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"

    def scale(self, s):
        return Vector(self.x * s, self.y * s)

    @property
    def length(self):
        return sqrt(self.x*self.x + self.y*self.y)

    def toNumpy(self):
        """
        Converti en array numpy
        """
        # TODO: sous optimal, devrait heriter de np.array plutot que creer des objets
        # convention numpy: height, width
        return np.array([self.y, self.x])

    def toInt(self):
        return Vector(int(self.x), int(self.y))

    def rot90(self, k):
        if k == 0:
            return Vector(self.x, self.y)

        if k == 1:
            # ccw
            return Vector(self.y, -self.x)

        if k == 2:
            # flip
            return Vector(-self.x, -self.y)

        if k == 3 or k == -1:
            # cw
            return Vector(-self.y, self.x)

        raise ValueError("Vector::rot90, invalide k")

    @staticmethod
    def fromNumpy(v):
        return Vector(v[1], v[0])

    @staticmethod
    def dot(a, b):
        return a.x * b.x + a.y * b.y

    @staticmethod
    def winding(a, b):
        """
        Retourne -1 si a tourne vers b de maniere CW
        Retourne  1 si a tourne vers b de maniere CCW
        Retourne  0 si a et b sont paralleles
        """
        k = a.x * b.y - a.y * b.x

        if k > 0:
            return -1

        if k < 0:
            return 1

        return 0
