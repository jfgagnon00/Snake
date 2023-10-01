from enum import IntEnum
from pygame import BLEND_RGBA_MULT
from pygame.rect import Rect
from pygame.sprite import Group


from game.Vector import Vector
from .Sprite import _Sprite


class _Edge(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4

class _SpriteType(IntEnum):
    TAIL = 0
    BODY = 1
    HEAD = 2
    EYES = 3

def _getEdgeFromDir(outDir):
    if outDir.x > 0:
        return _Edge.EAST

    if outDir.x < 0:
        return _Edge.WEST

    if outDir.y > 0:
        return _Edge.SOUTH

    if outDir.y < 0:
        return _Edge.NORTH

    return _Edge.NONE

def _getKey(spriteType, edge1, edge2):
        return int(spriteType) | (int(edge1) << 3) | (int(edge2) << 6)

class _GraphicSnake():
    def __init__(self, graphicsConfig, gridCellSize, offset):
        self._tileSize = gridCellSize
        self._offset = offset

        # le contenu de l'image du serpent suit des regles
        # implicites pour le positionement des parties du
        # corps; ouvrir dans gimp pour comprendre
        spriteSheet = _Sprite(filename=graphicsConfig.snakeSpritesPath)
        spriteSheet.optimize(True)

        # customisation du serpent
        w = graphicsConfig.snakeTileSize[0]
        h = graphicsConfig.snakeTileSize[1]
        spriteSheet.image.fill(graphicsConfig.snakeColor,
                               Rect(0, 0, 3 * w, 5 * h),
                               BLEND_RGBA_MULT)

        # remettre a l'echelle de la grille
        w = 4 * self._tileSize
        h = 5 * self._tileSize
        spriteSheet.resize((w, h))

        # extraite chaque sous image
        self._images = {}

        self._images[_getKey(_SpriteType.TAIL, _Edge.NONE, _Edge.WEST)] = self._getSubImage(spriteSheet, 0, 0)
        self._images[_getKey(_SpriteType.TAIL, _Edge.NONE, _Edge.NORTH)] = self._getSubImage(spriteSheet, 0, 1)
        self._images[_getKey(_SpriteType.TAIL, _Edge.NONE, _Edge.EAST)] = self._getSubImage(spriteSheet, 0, 2)
        self._images[_getKey(_SpriteType.TAIL, _Edge.NONE, _Edge.SOUTH)] = self._getSubImage(spriteSheet, 0, 3)

        self._images[_getKey(_SpriteType.BODY, _Edge.WEST, _Edge.EAST)] = self._getSubImage(spriteSheet, 1, 0)
        self._images[_getKey(_SpriteType.BODY, _Edge.WEST, _Edge.SOUTH)] = self._getSubImage(spriteSheet, 1, 1)
        self._images[_getKey(_SpriteType.BODY, _Edge.WEST, _Edge.NORTH)] = self._getSubImage(spriteSheet, 1, 2)
        self._images[_getKey(_SpriteType.BODY, _Edge.NORTH, _Edge.EAST)] = self._getSubImage(spriteSheet, 1, 3)
        self._images[_getKey(_SpriteType.BODY, _Edge.SOUTH, _Edge.EAST)] = self._getSubImage(spriteSheet, 1, 4)
        self._images[_getKey(_SpriteType.BODY, _Edge.NORTH, _Edge.SOUTH)] = self._getSubImage(spriteSheet, 2, 4)

        self._images[_getKey(_SpriteType.HEAD, _Edge.WEST, _Edge.NONE)] = self._getSubImage(spriteSheet, 2, 0)
        self._images[_getKey(_SpriteType.HEAD, _Edge.NORTH, _Edge.NONE)] = self._getSubImage(spriteSheet, 2, 1)
        self._images[_getKey(_SpriteType.HEAD, _Edge.EAST, _Edge.NONE)] = self._getSubImage(spriteSheet, 2, 2)
        self._images[_getKey(_SpriteType.HEAD, _Edge.SOUTH, _Edge.NONE)] = self._getSubImage(spriteSheet, 2, 3)

        self._images[_getKey(_SpriteType.EYES, _Edge.WEST, _Edge.NONE)] = self._getSubImage(spriteSheet, 3, 0)
        self._images[_getKey(_SpriteType.EYES, _Edge.NORTH, _Edge.NONE)] = self._getSubImage(spriteSheet, 3, 1)
        self._images[_getKey(_SpriteType.EYES, _Edge.EAST, _Edge.NONE)] = self._getSubImage(spriteSheet, 3, 2)
        self._images[_getKey(_SpriteType.EYES, _Edge.SOUTH, _Edge.NONE)] = self._getSubImage(spriteSheet, 3, 3)

        self._group = Group()
        self._sprites = []

    def update(self, gameSnake):
        gameCount = len(gameSnake.bodyParts)
        while (gameCount + 1) > len(self._sprites):
            self._sprites.append(_Sprite(x=0, y=0, w=self._tileSize, h=self._tileSize))

        # head et eyes
        p0 = gameSnake.bodyParts[0]
        p1 = gameSnake.bodyParts[1]
        inDir = p0 - p1

        p = self._environmentToWindow(p0)
        head = self._getImage(_SpriteType.HEAD, inDir, Vector(0, 0))
        self._updateSprite(0, p, head)

        eyes = self._getImage(_SpriteType.EYES, inDir, Vector(0, 0))
        self._updateSprite(1, p, eyes)

        # body parts
        spriteIndex = 2
        for i in range(1, gameCount - 1):
            outDir = inDir
            p0 = p1
            p1 = gameSnake.bodyParts[i + 1]
            inDir = p0 - p1

            p = self._environmentToWindow(p0)
            body = self._getImage(_SpriteType.BODY, inDir, outDir)
            self._updateSprite(spriteIndex, p, body)
            spriteIndex += 1

        # tail
        p = self._environmentToWindow(p1)
        tail = self._getImage(_SpriteType.TAIL, Vector(0, 0), -inDir)
        self._updateSprite(spriteIndex, p, tail)

        self._group.empty()
        self._group.add(self._sprites[:gameCount + 1])

    def render(self, image):
        self._group.draw(image)

    def _getSubImage(self, spriteSheet, x, y):
        x *= self._tileSize
        y *= self._tileSize
        return spriteSheet.image.subsurface( Rect(x, y, self._tileSize, self._tileSize) )

    def _getImage(self, spriteType, inDir, outDir):
        e0 = _getEdgeFromDir(-inDir)
        e1 = _getEdgeFromDir(outDir)

        key = _getKey(spriteType, e0, e1)
        if key in self._images:
            return self._images[key]

        key = _getKey(spriteType, e1, e0)
        return self._images[key]

    def _environmentToWindow(self, vector):
        # TODO: code duplique, refactorer
        return Vector(vector.x * self._tileSize,
                      vector.y * self._tileSize + self._offset)

    def _updateSprite(self, index, position, image):
        sprite = self._sprites[index]
        sprite.image = image
        sprite.rect.x = position.x
        sprite.rect.y = position.y
