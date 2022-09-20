#!/usr/bin/bash

pip install jupytext

pip install nibabel

mkdir -p ~/.virtualenvs/

python3 -m venv ~/.virtualenvs/carp-env
source ~/.virtualenvs/carp-env/bin/activate

pip install -r requirements.txt

sudo apt update -y

sudo apt install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra

