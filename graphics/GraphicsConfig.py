class GraphicsConfig():
    """
    Represente les configurations pour tout ce qui 
    correspond a l'affichage
    """
    def __init__(self):
        self.window_width = 400
        self.window_height = 400
        
        self.snakeSpritesPath = ""
        self.snakeSpritesSheet = {}

        self.foodSpritesPath = ""
        self.foodSpritesSheet = {}