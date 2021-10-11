from _parser import custom_parser, get_paths
from dask import delayed
import dask.array as da
from datetime import datetime
import glob
import json
import napari
from nd2reader import ND2Reader
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tensorstore as ts
from tensorstore import TensorStore
import toolz as tz
import zarr


# ND2 as Zarr
# -----------
def nd2_2_zarr(path):
    """
    Save ND2 data as a zarr (all channels) based on single path or inputted directory
    """
    if os.path.isdir(path):
        expr = os.path.join(path, '*.nd2')
        files = glob.glob(expr)
        for f in files:
            _nd2_2_zarr(os.path.join(path, f))
    else:
        _nd2_2_zarr(path)


def _nd2_2_zarr(data_path):
    """
    Save ND2 data as a zarr (all channels)
    """
    nd2_data = ND2Reader(data_path)
    meta = nd2_data.metadata
    input_path = Path(data_path)
    save_path = os.path.join(str(input_path.parent), 
                             str(input_path.stem) + '.zarr')
    if not os.path.exists(save_path):
        t = meta['num_frames']
        y = meta['height']
        x = meta['width']
        z = meta['z_levels'].stop
        n_channels  = len(meta['channels'])
        shape = (n_channels, t, y, x, z)
        # open a zarr array of correct size
        z = zarr.open_array(
                            save_path, 
                            mode='w', 
                            shape=shape,
                            chunks=(1, 1, None, None, None), dtype='i4', fill_value=0)
        # iterate through channels, producing a dask array, then saving each time
        for c in range(n_channels):
            arr = get_stack(nd2_data, c=c, t_max=t)
            for i in range(t):
                a = arr[i, ...].compute()
                z[c, i, ...] = a

        # save metadata
    save_path = os.path.join(str(input_path.parent), 
                             str(input_path.stem) + '.json')
    if not os.path.exists(save_path):
        md = dict_2_JSON_serializable(meta)
        with open(save_path, 'w') as outfile:
            json.dump(md, outfile, indent=4)                      


# Read ND2
# --------
def get_nd2_vol(nd2_data, c, frame):
    """
    Get single frame of ND2Reader object
    """
    nd2_data.default_coords['c']=c
    nd2_data.bundle_axes = ('y', 'x', 'z')
    #nd2_data.iter_axes = 'c'
    #v = nd2_data[c]
    v = nd2_data.get_frame(frame)
    v = np.array(v)
    return v


def get_stack(nd2_data, c=2, frame=0, t_max=193, w_shape=False):
    """
    Get a single channel of an ND2Reader object as dask stack
    """
    nd2vol = tz.curry(get_nd2_vol)
    fram = get_nd2_vol(nd2_data, c, frame)
    arr = da.stack(
        [da.from_delayed(delayed(nd2vol(nd2_data, c))(i),
         shape=fram.shape,
         dtype=fram.dtype)
         for  i in range(t_max)]
    )
    shape = [t_max, ]
    shape[1:] = fram.shape
    if w_shape:
        return arr, shape
    else:
        return arr


# Metadata to JSON
# ----------------

def dict_2_JSON_serializable(meta):
    """""
    Convert ND2 metadata dict items to JSON serialisable form

    Returns
    -------
    md: dict
        JSON serialisable dict

    Notes
    -----
    Type conversions
    - datetime: str
    - ndarray: list
    - range: list(range.stop)
    """""
    md = meta.copy()
    for key in meta.keys():
        i = meta[key]
        md = _fix_values(key, md)
    return md


def _fix_values(key, md):
    """
    Recursively search each item in metadata
    and reassign values.
    """
    i = md[key]
    md = _set_val(i, key, md)
    if isinstance(i, list):
        new = []
        for idx in range(len(i)):
            new.append(_fix_values(idx, i)[idx])
        md[key] = new
    if isinstance(i, dict):
        new = {}
        for k in i.keys():
            new[k] = _fix_values(k, i)[k]
        md[key] = new
    return md


def _set_val(i, key, md):
    """
    Reassign values, changing type where necessary.
    """
    if isinstance(i, datetime):
        md[key] = i.strftime("%m/%d/%Y, %H:%M:%S")
    elif isinstance(i, range):
        md[key] = [i.stop]
    elif isinstance(i, np.ndarray):
        md[key] = i.tolist()
    elif isinstance(i, pd.DataFrame):
        i = i.astype(int, errors='ignore')
        md[key] = i.to_dict()
    elif isinstance(i, slice):
        md[key] = [i.start, i.stop, i.step]
    elif type(i) in (np.int16, np.int32, np.int64, np.int8):
        md[key] = int(i)
        print(type(i))
    else:
        md[key] = i
    return md


# Zarr via dask
# ----------------------
def single_zarr(input_path, c=2, idx=0):
    '''
    Parameters
    ----------
    c: int or tuple
        Index of indices to return in array
    idx: int or tuple
        which indicies of the dim to apply c to
    '''
    assert type(c) == type(idx)
    arr = da.from_zarr(input_path)
    slices = [slice(None)] * arr.ndim
    if isinstance(idx, int):
        slices[idx] = c
    elif isinstance(idx, tuple):
        for i, ind in enumerate(idx):
            slices[ind] = c[i]
    else:
        raise TypeError('c and idx must be int or tuple with same type')
    slices  = tuple(slices)
    arr = arr[slices]
    return arr


def view_zarr(input_path, scale=(1, 1, 1, 1, 4)):
    arr = da.from_zarr(input_path)
    with napari.gui_qt():
        viewer = napari.Viewer() 
        viewer.add_image(arr, name='all_channels', scale=scale)


# Zarr via tensorstore
# --------------------

#def shape(tsobj):
    #open_spec = tsobj.spec().to_json()
    # Tensorstore input_exclusive_max may have mixed list and int elements
    #input_exc_max = flatten_list(open_spec['transform']['input_exclusive_max'], [])
    #input_exc_max = np.array(input_exc_max)
    #input_inc_min = np.array(open_spec['transform']['input_inclusive_min'])
    #s = input_exc_max - input_inc_min
    #return s


#def flatten_list(x, final):
    #"""
   # Tensorstore input_exclusive_max may have mixed lists
   # """
    #for item in x:
    #    if isinstance(item, list):
       #     flatten_list(item, final)
      #  else:
     #     final.append(item)
    #return final


#def ndim(tsobj):
   # return len(shape(tsobj))


#def hacky_ts_zarr_open(open_spec):
    #TensorStore.shape = property(shape)
    #TensorStore.ndim = property(ndim)
   # TensorStore.copy = TensorStore.__array__
    #arr = ts.open(open_spec, create=False, open=True).result()
   # return arr



# Save ND2 2 Zarr
# ---------------
if __name__ == "__main__":

    # Parser
    # ------
    parser = custom_parser()
    args = parser.parse_args()
    path = get_paths(args, 
                      'view_segmentation_3D', 
                      get={'data_path':'image'}, 
                      by_name=True
                      )['data_path'] # TODO: fix this crap
    # Save Zarrs
    # ----------
    nd2_2_zarr(path)

    
