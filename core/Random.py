"""
Encapsulation de random pour permettre override des methodes.
"""

import random

from abc import ABC, abstractmethod

class RandomProxy:
    # instance du proxy a utiliser
    instance = None

    @staticmethod
    def init():
        RandomProxy.instance = _DefaultRandom()

    @staticmethod
    def choice(*args):
        return RandomProxy.instance.choice(*args)

class RandomBase(ABC):
    """
    Interface a implementer pour l'override de random
    """
    @abstractmethod
    def choice(self, *args):
        pass

class _DefaultRandom(RandomBase):
    """
    Implementation par default pour RandomProxy
    """
    def choice(self, *args):
        return random.choice(*args)
