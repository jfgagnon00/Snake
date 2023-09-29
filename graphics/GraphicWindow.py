from pygame import BLEND_RGBA_MULT
from pygame.display import set_mode, flip
from pygame.rect import Rect
from pygame.sprite import Group
from pygame.surface import Surface
from pygame.time import Clock

from .Sprite import Sprite

class GraphicWindow():
    """
    Encapsule les objets pygame pour l'affichage et
    l'animation de GameEnvironment
    """
    def __init__(self, simulationGridShape, graphicsConfig):
        self._clock = Clock()
        self._fps = graphicsConfig.fps
        self._clearColor = graphicsConfig.clearColor

        # attention #1: simulationGridShape utlise la convention (w, h)
        # attention #2: pygame aime bien les coordonnes en pixels, attention aux
        #               operations en nombres flotant
        aspectRatio = simulationGridShape[0] / simulationGridShape[1]
        if aspectRatio >= 1:
            self._tileSize = int(graphicsConfig.windowSize / simulationGridShape[0])
        else:
            self._tileSize = int(graphicsConfig.windowSize / simulationGridShape[1])
        self._initBackgroundTiles(simulationGridShape, graphicsConfig)

        w = simulationGridShape[0] * self._tileSize
        h = simulationGridShape[1] * self._tileSize
        self._canvas = set_mode((w, h))

        self._initFood(graphicsConfig)
        self._initSnake(graphicsConfig)

    def update(gameEnvironment):
        pass

    def render(self, ):
        self._canvas.fill(self._clearColor)
        self._backgroundTiles.draw(self._canvas)
        self._food.draw(self._canvas)
        self._snake.draw(self._canvas)

    def flip(self):
        flip()
        self._clock.tick(self._fps)

    def _initBackgroundTiles(self, simulationGridShape, graphicsConfig):
        self._backgroundTiles = Group()

        tileShape = (self._tileSize, self._tileSize)
        tileLight = Surface(tileShape)
        tileLight.fill(graphicsConfig.backgroundTileColorLight)
        tileDark = Surface(tileShape)
        tileDark.fill(graphicsConfig.backgroundTileColorDark)
        tileSurfaces = (tileLight, tileDark)

        rowTileSurfaceIndex = 0
        y = 0
        for _ in range(simulationGridShape[1]):
            tileSurfaceIndex = rowTileSurfaceIndex
            x = 0
            for _ in range(simulationGridShape[0]):
                sprite = Sprite(image=tileSurfaces[tileSurfaceIndex])
                sprite.rect.x = x
                sprite.rect.y = y

                self._backgroundTiles.add(sprite)

                tileSurfaceIndex = 1 - tileSurfaceIndex
                x += self._tileSize

            rowTileSurfaceIndex = 1 - rowTileSurfaceIndex
            y += self._tileSize

    def _initFood(self, graphicsConfig):
        self._foodSprite = Sprite(filename=graphicsConfig.foodSpritePath)
        self._foodSprite.optimize(True)
        self._foodSprite.resize((self._tileSize, self._tileSize))
        self._food = Group(self._foodSprite)

    def _initSnake(self, graphicsConfig):
        # le contenue du sprite du serpent suit des regles
        # implicites pour le positionement des parties du
        # corps; ouvrir dans gimp pour comprendre
        snake = Sprite(filename=graphicsConfig.snakeSpritesPath)
        snake.optimize(True)

        # customisation du serpent
        w = graphicsConfig.snakeTileSize[0]
        h = graphicsConfig.snakeTileSize[1]
        snake.image.fill(graphicsConfig.snakeColor,
                         Rect(0, 0, 3 * w, 5 * h),
                         BLEND_RGBA_MULT)

        # remettre a l'echelle du canvas
        w = 4 * self._tileSize
        h = 5 * self._tileSize
        snake.resize((w, h))

        # decouper en sous sprites
        self._snakeTails = []
        self._snakeTails.append( self._getSnakeSubSprite(snake, 0, 0) )
        self._snakeTails.append( self._getSnakeSubSprite(snake, 0, 1) )
        self._snakeTails.append( self._getSnakeSubSprite(snake, 0, 2) )
        self._snakeTails.append( self._getSnakeSubSprite(snake, 0, 3) )

        self._snakeBodies = []
        self._snakeBodies.append( self._getSnakeSubSprite(snake, 1, 0) )
        self._snakeBodies.append( self._getSnakeSubSprite(snake, 1, 1) )
        self._snakeBodies.append( self._getSnakeSubSprite(snake, 1, 2) )
        self._snakeBodies.append( self._getSnakeSubSprite(snake, 1, 3) )
        self._snakeBodies.append( self._getSnakeSubSprite(snake, 1, 4) )
        self._snakeBodies.append( self._getSnakeSubSprite(snake, 2, 4) )

        self._snakeHeads = []
        self._snakeHeads.append( self._getSnakeSubSprite(snake, 2, 0) )
        self._snakeHeads.append( self._getSnakeSubSprite(snake, 2, 1) )
        self._snakeHeads.append( self._getSnakeSubSprite(snake, 2, 2) )
        self._snakeHeads.append( self._getSnakeSubSprite(snake, 2, 3) )

        self._snakeEyes = []
        self._snakeEyes.append( self._getSnakeSubSprite(snake, 3, 0) )
        self._snakeEyes.append( self._getSnakeSubSprite(snake, 3, 1) )
        self._snakeEyes.append( self._getSnakeSubSprite(snake, 3, 2) )
        self._snakeEyes.append( self._getSnakeSubSprite(snake, 3, 3) )

        self._snake = Group(self._snakeHeads[0], self._snakeEyes[0])

    def _getSnakeSubSprite(self, sprite, x, y):
        x *= self._tileSize
        y *= self._tileSize
        image = sprite.image.subsurface( Rect(x, y, self._tileSize, self._tileSize) )
        return Sprite(image=image)
