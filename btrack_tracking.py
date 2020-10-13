import btrack
from btrack.constants import BayesianUpdates
from btrack.dataio import localizations_to_objects
import napari
import numpy as np
import os
import pandas as pd
from pathlib import Path
from parser import custom_parser, hardcoded_paths
from view_tracks import get_stack, get_tracks, base, get_paths

local_dir = os.path.realpath(__file__)
local_dir = str(Path(local_dir).root)

def convert_to_df(tracks):
    to_concat = []
    for i in range(len(tracks)):
        my_tracks = tracks[i].to_dict()
        df = pd.DataFrame(my_tracks)
        to_concat.append(df)
    out = pd.concat(to_concat)
    return out


def read_to_track(
                  df, 
                  t_x_y_z_id=['frame', 'x', 'y', 'z', 'pid'], 
                  z_scale=4, 
                  config_name='platelet_config.json', 
                  local_dir=local_dir
                  ):
    data_df = df[t_x_y_z_id]
    t = t_x_y_z_id[0]
    ID = t_x_y_z_id[4]
    data_df = data_df.rename(columns={t:'t', ID:'ID'})
    data_df['z'] = data_df['z'] * z_scale
    objects_to_track = localizations_to_objects(data_df)
    config_path = os.path.join(local_dir, config_name)
    return objects_to_track, config_path


def track(
          objects_to_track, 
          config_path, 
          shape,
          max_search_radius=100, 
          step_size=100
          ):
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure_from_file(config_path)
        # set the update method and maximum search radius (both optional)
        tracker.update_method = BayesianUpdates.EXACT
        tracker.max_search_radius = max_search_radius
        # append the objects to be tracked
        tracker.append(objects_to_track)
        # set the volume (Z axis volume is set very large for 2D data)
        tracker.volume=((0, shape[2]),(0, shape[1]),(0, shape[0]))
        # track them (in interactive mode)
        tracker.track_interactive(step_size=step_size)
        # generate hypotheses and run the global optimiser
        tracker.optimize()
        # get the tracks as a python list
        tracks = tracker.tracks
    return tracks


if __name__ == "__main__":

    # Parser
    # ------
    parser = custom_parser(coords=True, save=True, base=base)
    args = parser.parse_args()
    paths = get_paths(args, 
                      __file__,
                      get={'data_path':'image', 
                           'coords_path':'coords',
                           'save_path':'save' 
                           })
      
    # Data
    # ----
    arr, shape = get_stack(paths['data_path'], w_shape=True)
    df = pd.read_csv(paths['coords_path'])

    # Btrack io
    # ---------
    objects_to_track, config_path = read_to_track(df)

    # Tracking
    # --------
    tracks = track(objects_to_track, config_path, shape, max_search_radius=25)
    tracks_df = convert_to_df(tracks)
    tracks_df.to_csv(paths['save_path'])
    tracks = get_tracks(tracks_df, id_col='parent', time_col='t', w_prop=False)


    # Visualise image and tracks
    # --------------------------
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(arr, scale=[1, 1, 1, 4])
        viewer.add_tracks(tracks, colormap='viridis')
