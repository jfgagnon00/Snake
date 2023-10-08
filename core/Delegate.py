
class Delegate():
    """
    Encapsule le design pattern delegation. Enregistre une
    liste de callable qui seront notifies lors de l'appel
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
        """
        Active le delegate. Responsabilite des appelants de register/unregister
        de determiner l'ordre et de maintenir la coherance avec args
        """
        for callable in self._delegates:
            callable(*args)
