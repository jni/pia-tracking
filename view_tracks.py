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


def get_stack(data_path, object_channel=2, frame=70):
    nd2_data = ND2Reader(data_path)
    nd2vol = tz.curry(get_nd2_vol)
    fram = get_nd2_vol(nd2_data, object_channel, frame)
    arr = da.stack(
        [da.from_delayed(delayed(nd2vol(nd2_data, 2))(i),
         shape=fram.shape,
         dtype=fram.dtype)
         for  i in range(193)]
    )
    return arr


def get_tracks(df, min_frames=20, id_col='particle', time_col='frame',
        coord_cols=('x', 'y', 'z'), log=False, scale=(1, 1, 1)):
    """
    Get the tracks for napari tracks. Will need to be updated before 
    use with recent commits to the tracks PR (ID + data)
    """
    time_0 = time.time()
    num_cols = len(coord_cols) + 2
    id_array = df[id_col].to_numpy()
    track_count = np.bincount(id_array)
    df['track length'] = track_count[id_array]
    df_filtered = df.loc[df['track length'] >= min_frames, :]
    data_cols = [id_col, time_col] + coord_cols
    track_data = df_filtered[data_cols].to_numpy()
    track_data[:, -3:] *= scale
    print(f'{np.sum(track_count >= min_frames)} tracks found in '
          f'{time.time() - time_0} seconds')
    return track_data, dict(df_filtered)


def save_tracks(tracks, name='tracks-for-napari.txt'):
    '''
    For saving tracks in old napari tracks format
    Not useful once changed to ID + data
    '''
    output = []
    for track in tracks:
        ls = track.tolist()
        output.append(ls)
    with open(name, 'w+') as f:
        f.write(str(output))


if __name__ == '__main__':
    # Construct Parser
    # ----------------
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
    parser = custom_parser(tracks=True, base=base)
    args = parser.parse_args()

    # Sortcuts or No
    # --------------
    if args.name:
        paths = hardcoded_paths(args.name, __file__)
    else:
        paths = {
            'data_path' : args.image, 
            'tracks_path' : args.tracks
        }

    # Get Data
    # --------
    arr = get_stack(paths['data_path'])
    df = pd.read_csv(paths['tracks_path'])
    if args.min_frames:
        tracks = get_tracks(df, scale=[1, 1, 4], min_frames=args.min_frames)
    else:
        tracks = get_tracks(df, scale=[1, 1, 4])
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
