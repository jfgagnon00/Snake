from enum import IntEnum

class GridOccupancy(IntEnum):
    """
    Etats possibles de chaque cellule de la grille de simulation
    """
    EMPTY = 0
    SNAKE_TAIL = 32
    SNAKE_BODY = 64
    SNAKE_HEAD = 128
    FOOD = 255
