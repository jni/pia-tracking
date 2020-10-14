# create and activate the environment 'platelet-tracking'
conda env create -f ./environment.yml
conda activate platelet-tracking
# install development version of napari
git clone https://github.com/napari/napari.git
cd napari
pip install -e .[all]