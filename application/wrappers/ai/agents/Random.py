"""
Utilitaires pour RandomProxy pour l'enregistrement/playback
"""

from core import RandomBase


class _RandomRecorder(RandomBase):
    def __init__(self, instance):
        super().__init__()
        self._instance = instance
        self.reset()

    @property
    def choices(self):
        return self._choices

    def reset(self):
        self._choices = []

    def choice(self, *args):
        choice_ = self._instance.choice(*args)
        self._choices.append(choice_)
        return choice_

class _RandomPlayback(RandomBase):
    def __init__(self, choices):
        super().__init__()
        self._choices = choices
        self.reset()

    def reset(self):
        self._next = 0

    def choice(self, *args):
        choice_ = self._choices[self._next]
        self._next += 1
        return choice_
