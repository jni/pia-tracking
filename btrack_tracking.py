import btrack
from btrack.constants import BayesianUpdates
from btrack.dataio import localizations_to_objects
import napari
import numpy as np
import os
import pandas as pd
from pathlib import Path
from parser import custom_parser, hardcoded_paths
from view_tracks import get_stack, get_tracks, base, shortcuts_or_no


def convert_to_df(tracks):
    to_concat = []
    for i in range(len(tracks)):
        my_tracks = tracks[i].to_dict()
        df = pd.DataFrame(my_tracks)
        to_concat.append(df)
    out = pd.concat(to_concat)
    return out


# Parser
# ------
parser = custom_parser(tracks=True, base=base)
args_ = parser.parse_args()
paths = shortcuts_or_no(args_)

# Data
# ----
arr, shape = get_stack(paths['data_path'], w_shape=True)
df = pd.read_csv(paths['tracks_path'])

# Btrack io
# ---------
data_df = df[['frame', 'x', 'y', 'z', 'pid']]
data_df = data_df.rename(columns={'frame':'t', 'pid':'ID'})
data_df['z'] = data_df['z'] * 4
print(data_df.head())
objects_to_track = localizations_to_objects(data_df)
config_path = os.path.realpath(__file__)
config_path = os.path.join(str(Path(config_path).root), 'platelet_config.json')

# Tracking
# --------
with btrack.BayesianTracker() as tracker:
  # configure the tracker using a config file
  tracker.configure_from_file(config_path)
  # set the update method and maximum search radius (both optional)
  tracker.update_method = BayesianUpdates.EXACT
  tracker.max_search_radius = 100
  # append the objects to be tracked
  tracker.append(objects_to_track)
  # set the volume (Z axis volume is set very large for 2D data)
  tracker.volume=((0, shape[2]),(0, shape[1]),(0, shape[0]))
  # track them (in interactive mode)
  tracker.track_interactive(step_size=100)
  # generate hypotheses and run the global optimiser
  tracker.optimize()
  # get the tracks as a python list
  tracks = tracker.tracks

tracks_df = convert_to_df(tracks)
save = hardcoded_paths(args_.name, __file__)
tracks_df.to_csv(save)
tracks = get_tracks(tracks_df, id_col='parent', time_col='t', w_prop=False)


# Visualise image and tracks
# --------------------------
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(arr, scale=[1, 1, 1, 4])
    viewer.add_tracks(tracks, colormap='viridis')
