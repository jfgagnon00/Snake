class GraphicsConfig():
    """
    Represente les configurations pour tout ce qui
    correspond a l'affichage
    """
    def __init__(self):
        # couleur de background
        self.clearColor = (255, 0, 255)

        # couleur des tiles des background
        self.backgroundTileColorLight = (0, 255, 0)
        self.backgroundTileColorDark = (0, 128, 0)

        # frames par seconde
        # 0 signifie infinie
        self.fps = 0

        # dimension du cote le plus large de la fenetre en pixels
        # le rapport d'aspect est deduit par la taille de la grille de simulation
        self.windowSize = 512

        # taille du texte en point
        self.fontSize = 12
        self.fontColor = (255, 255, 255)
        self.fontPath = ""

        # proprietes pour le serpents
        self.snakeSpritesPath = ""
        self.snakeColor = (50, 50, 200)
        self.snakeTileSize = (64, 64)

        # proprietes pour la nouriture
        self.foodSpritePath = ""
