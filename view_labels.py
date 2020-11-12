# from _parser import custom_parser, get_paths
import dask.array as da
import glob 
import napari
import os
from pathlib import Path

def view_segmentations(data_path, channel=2, scale=(1, 1, 1, 4)):
    """
    Given a directory, view all labels files with images
    """
    label_files = glob.glob(os.path.join(data_path, '*_labels.zarr'))
    paths = []
    for f in label_files:
        paths.append(f)
        f = Path(f)
        name = f.stem
        img_file = glob.glob(os.path.join(data_path, name[:name.find('_labels')] + '.zarr'))
        try: 
            img_file = img_file[0]
            paths.append(img_file)
        except IndexError:
            print(f"Cannot find an image corresponding to {name}")
    _view_from_paths(paths, scale)
    

def _view_from_paths(paths, scale):
    """
    View labels and image volumes from a list of paths
    """
    with napari.gui_qt():
        viewer = napari.Viewer()
        for path in paths:
            name = Path(path).stem
            array = da.from_zarr(path)
            if path.find('_labels.zarr') != -1:
                viewer.add_labels(
                                  array, 
                                  name=name, 
                                  scale=scale, 
                                  blending='additive', 
                                  visible=False
                                  )
            else:
                viewer.add_image(
                                 array, 
                                 name=name, 
                                 scale=scale, 
                                 blending='additive', 
                                 visible=False
                                 )

# this requires updates to _parser.py
# if __name__ == '__main__':
    # Parser
