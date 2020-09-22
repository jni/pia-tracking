# based on https://gist.github.com/AbigailMcGovern/3ec44eb509d4c8740248e1322f0e33d1
# needs https://github.com/napari/napari/pull/1361
from dask import delayed
import dask.array as da
import napari
from nd2reader import ND2Reader
import numpy as np
import os
import pandas as pd
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


def get_tracks(df, min_frames=20, id_col='particle', time_col='frame',
        coord_cols=['x', 'y', 'z'], log=False, scale=(1, 1, 1)):
    time_0 = time.time()
    num_cols = len(coord_cols) + 1
    track_ids = df[id_col].unique()
    tracks = []
    for id_ in track_ids:
        t0 = time.time()
        id_df = df[df[id_col]==id_]
        frames = len(id_df)
        if frames >= min_frames:
            track = np.zeros((frames, num_cols), dtype=np.float)
            t = np.array(id_df[time_col].values).T
            track[:, 0] = t
            for i, col in enumerate(coord_cols):
                coord = np.array(id_df[col].values).T * scale[i]
                track[:, i + 1] = coord
            tracks.append(track)
            t1 = time.time()
            if log:
                with open('log.txt', 'a+') as file_:
                    file_.write(f'Particle: {id_}, frames: {frames}, in {t1-t0} seconds\n')
    time_ = time.time() - time_0
    print(f'{len(tracks)} tracks found in {time_} seconds')
    if log:
        with open('log.txt', 'a+') as file_:
                file_.write(f'Total tracks: {len(tracks)}, Total time: {time_} seconds\n')
    return tracks


def save_tracks(tracks, name='tracks-for-napari.txt'):
    output = []
    for track in tracks:
        ls = track.tolist()
        output.append(ls)
    with open(name, 'w+') as f:
        f.write(str(output))


if __name__ == '__main__':
    # Image Data
    # ----------
    data_path = (
        '/Users/jni/Dropbox/share-files/200519_IVMTR69_Inj4_dmso_exp3.nd2'
    )
    nd2_data = ND2Reader(data_path)
    object_channel = 2
    nd2vol = tz.curry(get_nd2_vol)
    fram = get_nd2_vol(nd2_data, object_channel, 70)
    arr = da.stack(
        [da.from_delayed(delayed(nd2vol(nd2_data, 2))(i),
         shape=fram.shape,
         dtype=fram.dtype)
         for  i in range(193)]
    )


    # Tracks data
    # -----------
    path = '/Users/jni/Dropbox/share-files/tracks.csv'
    df = pd.read_csv(path)
    tracks = get_tracks(df, scale=[1, 1, 4])
    # save_path = '/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/tracks-for-napari.txt'
    # save_tracks(tracks, save_path)


    # Visualise image and tracks
    # --------------------------
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(arr, scale=[1, 1, 1, 4])
        viewer.add_tracks(tracks) #, scale=[1, 1, 1, 4])
