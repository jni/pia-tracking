# Scripts/notebooks to segment and track platelets in a growing thrombus

Current files:

- `fl.py` contains some utilities used in the notebooks, including objects for
  opening the ND2 file and keeping track of various processed data products
  (smoothing, thresholding, segmentation, etc.)
- The "process images" notebook is the first step, goes from images to
  segmentation. It creates output files in a timestamped subfolder of the
  current working directory.
- The "define variables" notebook then uses that segmentation to create tracks.

Goals:

- turn this from a set of loosely tied notebooks to a command-line utility that
  runs various parts of the pipeline
- use zarr to save the original data and intermediate products
- use dask for processing where possible
- get the tracks into the napari tracks layer from napari/napari#1361
