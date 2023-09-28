"""
Demarrer le jeu en mode interactif
"""


import os

from configs.configs import createConfigs


if __name__ == "__main__":
    # mettre le repoire courant sur le repertoire 
    # contenant ce fichier (facilite la gestion des path relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    configs = createConfigs("config_overrides.json")

    print(configs)