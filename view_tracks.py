# based on https://gist.github.com/AbigailMcGovern/3ec44eb509d4c8740248e1322f0e33d1
# needs https://github.com/napari/napari/pull/1361
from dask import delayed
import dask.array as da
import napari
import numpy as np
import os
import pandas as pd
from _parser import custom_parser, get_paths, track_view_base
import time
from data_io import single_zarr


# Functions
# ---------

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



if __name__ == '__main__':
    # Construct Parser
    # ----------------
    # construct dict with information for command line argument to 
    # interact with min_frames in get_tracks
    parser = custom_parser(tracks=True, base=track_view_base)
    args_ = parser.parse_args()
    paths = get_paths(args_, __file__)

    # Get Data
    # --------
    arr = single_zarr(paths['data_path'])
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
