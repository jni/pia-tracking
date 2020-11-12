# for iPython

from _parser import shortcuts
from annotate_tracks import CostEvaluation
from btrack_tracking import track_objects
from data_io import single_zarr
import fl
import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from view_tracks import get_tracks


# Data
# ----
data_path = shortcuts['Abi']['view_segmentation_3D']['data_path']

# Segmentation
# ------------

df_files = fl.nd2_info_to_df(data_path)

conf = dict(
            process_type = 'multi_thread', # single_thread, multi_process, multi_thread
            multi_workers = 7, # dont use too many
            object_channel = 2,
            intensity_channels = [0, 1],
            dog_sigma1 = 1.7,
            dog_sigma2 = 2.0,
            threshold = 0.15,
            peak_min_dist = 3,
            z_dist = 2,
            center_roi = True,
            rotate = True,
            rotate_angle = 45,
            )

ivmObjects = fl.IvmObjects(conf)

# loop for processing multiple nd2-files
now_start = fl.get_datetime()
time = Path(fl.get_datetime())
time.mkdir(exist_ok=True)

for fileId in range(len(df_files)):
    # process file
    file, frames = df_files.iloc[fileId][['file', 't']]
    ivmObjects.add_nd2info(df_files.iloc[fileId]) #add nd2-file info to conf
    df_obj = ivmObjects.process_file(file, range(frames))#frames))
    
    #--------------------------------------------------------
    #Niklas changed this section to change name of file and directory
    file_path=Path(file)
    now = fl.get_datetime()
    df_filename = f'./{now_start}/{file_path.stem}.{now}.df.pkl'
    conf_filename = f'./{now_start}/{file_path.stem}.{now}.conf.yml'
    
    #--------------------------------------------------------
    # save result
    df_obj.to_pickle(df_filename)
    fl.save_yaml(conf, conf_filename)

d = '/Users/amcg0011/GitRepos/pia-tracking/20201110-105024'
df_obj.to_csv(os.path.join(d, 'all-objects.csv'))

# Btrack for each file
# --------------------
local_dir = os.getcwd()
paths = glob.glob(os.path.join(d, '*.pkl')) 
import btrack_tracking
import importlib
importlib.reload(btrack_tracking)

dfs = {}
for path in paths:
    name = Path(path).stem
    name = name[:name.find('.2020')]
    df = pd.read_pickle(path)
    save_name = os.path.join(d, name+'_objs.csv')
    df.to_csv(save_name)
    zarr_path = os.path.join(data_path, name+'.zarr')
    print(zarr_path)
    shape = single_zarr(zarr_path).shape
    tracks_df = btrack_tracking.track_objects(df, shape, local_dir=local_dir)
    save_name = os.path.join(d, name+'_btracks.csv')
    tracks_df.to_csv(save_name)
    # add to df so the object is acessible from ipython
    dfs[name] = {'obsj' : df, 'tracks': tracks_df}


# View tracks
# -----------

for name in dfs.keys():
    zarr_path = os.path.join(data_path, name+'.zarr')
    img = single_zarr(zarr_path)
    dfs[name]['image'] = img

import napari
with napari.gui_qt():
    viewer = napari.Viewer()
    for name in dfs.keys():
        df = dfs[name]['tracks']
        tracks = get_tracks(df, id_col='parent', time_col='t', w_prop=False)
        viewer.add_image(dfs[name]['image'], name=name, scale=(1, 1, 1, 4))
        viewer.add_tracks(tracks, name=name+'_trk')

