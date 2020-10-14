#!/bin/bash
CONDA_BASE=$(conda info --base)
shell="${CONDA_BASE}/etc/profile.d/conda.sh"
source $shell
conda deactivate
conda env remove -n platelet-tracking
rm -rf napari