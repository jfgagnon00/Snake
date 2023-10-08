
class Delegate():
    """
    Encapsule le design pattern delegation. Enregistre une
    liste de callable qui seront notifie lors de l'appel
    de ce delegate.
    """

    def __init__(self):
        self._delegates = []

    def register(self, callable):
        """
        Ajoute callable dans la liste des delegates
        """
        self._delegates.append(callable)

    def unregister(self, callable):
        """
        Eleve callable dans la liste des delegates
        """
        self._delegates.remove(callable)

    def clear(self):
        """
        Vide la liste des delegates
        """
        self._delegates.clear()

    def __call__(self, *args):
        for callable in self._delegates:
            callable(*args)
