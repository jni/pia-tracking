import btrack
from btrack.constants import BayesianUpdates
from btrack.dataio import localizations_to_objects
import numpy as np
import os
import pandas as pd
from pathlib import Path
from parser import custom_parser
from view_tracks import get_stack, get_tracks, base, shortcuts_or_no


# Parser
# ------
parser = custom_parser(tracks=True, base=base)
args_ = parser.parse_args()
paths = shortcuts_or_no(args_)

# Data
# ----
arr = get_stack(paths['data_path'])
df = pd.read_csv(paths['tracks_path'])

# Btrack io
# ---------
data_df = df[['frame', 'x', 'y', 'z', 'pid']]
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
  tracker.volume=((0,1200),(0,1600),(-1e5,1e5))
  # track them (in interactive mode)
  tracker.track_interactive(step_size=100)
  # generate hypotheses and run the global optimiser
  tracker.optimize()
  # get the tracks as a python list
  tracks = tracker.tracks
  box = tracker.volume