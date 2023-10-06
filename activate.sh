#!/bin/bash

# verifier si deja dans virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]
then
  InVenv=1
else
  InVenv=0
fi

if [ $InVenv -eq 1 ]; then
    echo "Dejà dans un environment python"
    return
fi

# verifier si snake virtual environment existe
NeedInstall=0
if ! [ -f .venv/bin/activate ]; then
    NeedInstall=1
    echo "Création virtual environment pour Snake"
    python3 -m venv .venv --prompt Snake
fi

echo "Activation virtual environment pour Snake"
source .venv/bin/activate

if [ $NeedInstall -eq 1 ]; then
    echo "Installation des dépendences"
    pip3 install -r requirements.txt
fi
