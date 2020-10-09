# based on https://gist.github.com/AbigailMcGovern/3ec44eb509d4c8740248e1322f0e33d1
# needs https://github.com/napari/napari/pull/1361
from dask import delayed
import dask.array as da
import napari
from nd2reader import ND2Reader
import numpy as np
import os
import pandas as pd
from parser import custom_parser, hardcoded_paths
import time
import toolz as tz


# Functions
# ---------
def get_nd2_vol(nd2_data, c, frame):
    nd2_data.default_coords['c']=c
    nd2_data.bundle_axes = ('y', 'x', 'z')
    v = nd2_data.get_frame(frame)
    v = np.array(v)
    return v


def get_stack(data_path, object_channel=2, frame=70, t_max=193, w_shape=False):
    nd2_data = ND2Reader(data_path)
    nd2vol = tz.curry(get_nd2_vol)
    fram = get_nd2_vol(nd2_data, object_channel, frame)
    arr = da.stack(
        [da.from_delayed(delayed(nd2vol(nd2_data, 2))(i),
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


def get_tracks(df, min_frames=20, id_col='particle', time_col='frame',
        coord_cols=('x', 'y', 'z'), scale=(1, 1, 1), w_prop=True):
    """
    Get the tracks from pandas.DataFrame containing object ID, 
    time, and coordinates for viewing in napari. Filters tracks
    according to a specified or default minimum number of frames 
    in which they should appear. Returns tuple containing an
    array with columns ID, t, <coord cols> and the properties
    dictionary.

    Parameters
    ----------
    df: pd.DataFrame
        tracks data
    min_frames: int
        minimum frames in which an object should appear
        to be added to the tracks data for viewing.
    id_col: str
        The name of the column in which object ID is found
    time_col: str
        The name of the column in which the time is found
    coord_cols: tuple or list of str
        names of columns containing the object coordinates
    scale: numeric or tuple or string of numeric
        To scale coordinates for viewing with the data
    
    Returns
    -------
    track_data: np.ndarray
        array with cols ID, t, <coords>
        The data is sorted by ID then t in accordance
        with napari tracks data validation
    dict(df_filtered): dict
        dict containing the properties for napari
    """
    time_0 = time.time()
    id_array = df[id_col].to_numpy()
    track_count = np.bincount(id_array)
    df['track length'] = track_count[id_array]
    df_filtered = df.loc[df['track length'] >= min_frames, :]
    df_filtered = df_filtered.sort_values(by=[id_col, time_col])
    data_cols = [id_col, time_col] + list(coord_cols)
    track_data = df_filtered[data_cols].to_numpy()
    track_data[:, -3:] *= scale
    print(f'{np.sum(track_count >= min_frames)} tracks found in '
          f'{time.time() - time_0} seconds')
    if w_prop:
        return track_data, dict(df_filtered)
    else:
        return track_data


def save_tracks(tracks, name='tracks-for-napari.csv'):
    '''
    For saving tracks into a csv file
    '''
    np.savetxt(name, tracks, delimiter=',')


def shortcuts_or_no(args_):
    if args_.name:
        paths = hardcoded_paths(args_.name, __file__)
    else:
        paths = {
            'data_path' : args_.image, 
            'tracks_path' : args_.tracks
        }
    return paths

# construct dict with information for command line argument to 
# interact with min_frames in get_tracks
h0 = "Minimum frames in which particles must appear to be "
h1 = "visualised as tracks (default = 20)"
base = {
        'min_frames' : {
            'name' :'--min_frames',
            'help' : h0 + h1, 
            'type' : int, 
            'default' : 20
        }
}

if __name__ == '__main__':
    # Construct Parser
    # ----------------
    # construct dict with information for command line argument to 
    # interact with min_frames in get_tracks
    parser = custom_parser(tracks=True, base=base)
    args_ = parser.parse_args()
    paths = shortcuts_or_no(args_)

    # Get Data
    # --------
    arr = get_stack(paths['data_path'])
    df = pd.read_csv(paths['tracks_path'])
    if args_.min_frames: # this if else block is actually unneccessary, 
        # apparently I deeply distrust argparse's default =P
        tracks, properties = get_tracks(df, scale=[1, 1, 4], 
                                        min_frames=args_.min_frames)
    else:
        tracks, properties = get_tracks(df, scale=[1, 1, 4])
    # save_path = '/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/tracks-for-napari.txt'
    # save_tracks(tracks, save_path)


    # Visualise image and tracks
    # --------------------------
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(arr, scale=[1, 1, 1, 4])
        viewer.add_tracks(
            tracks,
            properties=properties,
            color_by='particle',
            colormap='viridis',
        )
