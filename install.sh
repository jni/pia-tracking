#!/bin/bash
# create the environment 'platelet-tracking'
conda env create -f ./environment.yml
# functions are exported by default to subshells 
# for me, conda activate didn't run from this script
# used recomendations from https://github.com/conda/conda/issues/7980
CONDA_BASE=$(conda info --base)
shell="${CONDA_BASE}/etc/profile.d/conda.sh"
source $shell
conda activate platelet-tracking
# install development version of napari
pip install -r requirements.txt
git clone https://github.com/napari/napari.git
cd napari
pip install -e .[all]