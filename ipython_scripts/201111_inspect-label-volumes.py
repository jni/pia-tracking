from _parser import shortcuts
from data_io import single_zarr
import importlib
import fl
import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Data
# ----
data_path = shortcuts['Abi']['view_segmentation_3D']['data_path']

# Segmentation
# ------------
# NOTE: This was run after amendments were made to fl.py

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


