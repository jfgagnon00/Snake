class SpriteSubSurface():
    """
    Represente plusieurs regions d'une image. Utile pour animaiton
    ou joindre plusieurs images dans un seul fichier
    """
    def __init__(self):
        # dictionnaire de nom: Rect2D
        self._subSurfaces = {}
