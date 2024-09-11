<div align="center">
  <h1>Snake</h1>
  <img src="screenshot.png" width="512">
</div>

Projet fait dans le but d'explorer l'apprentissage par renforcement. Implémentation d'un agent pour le jeu de serpent.

# Installation

1. Cloner ce dépot.

```
git clone https://github.com/jfgagnon00/Snake.git
```

2. Installer python 3.11 ou supérieur

3. A la ligne de commande:

```
source activate.sh
```

Ou créer un manuellement un environment avec venv:

```
python3 -m venv .venv --prompt Snake
source .venv/bin/activate
pip3 install -r requirements.txt
```

# Utilisation

Snake utilise la ligne de commande pour exposer ses fonctionalités. Les commandes sont simples et auto documntées. A la ligne de commande, pour jouer:

```
python -m snake play --help
```

Pour entraînement:

```
python -m snake train --help
```

Aide en génrale:

```
python -m snake --help
```