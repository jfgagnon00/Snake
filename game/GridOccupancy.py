from enum import IntEnum

class GridOccupancy(IntEnum):
    """
    Etats possibles de chaque cellule de la grille de simulation
    """
    # doit etre la plus petite valeur
    EMPTY = 0
    SNAKE_BODY = 1
    SNAKE_HEAD = 2
    # doit etre la plus grande valeur
    FOOD = 3
