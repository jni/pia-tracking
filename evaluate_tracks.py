from _parser import custom_parser, get_paths, track_view_base
from annotate_tracks import annotate
import os
from random_tracks_engine import get_random_tracks, read_data


# argparse parse arguments 
#    - image data + tracks data + output directory 
parser = custom_parser(image=True, tracks=True, save=True)
args = parser.parse_args()
paths = get_paths(
                  args, 
                  'random_tracks_engine',
                  get={'data_path':'image', 
                       'tracks_path':'tracks',
                       'save_dir':'save', 
                       }, 
                  by_name=True
                  )
# get random tracks from tracks and volume
prefix = 'weighted_random_tracks'
arr, tracks, df = get_random_tracks(paths, prefix, n=15)
# save annotation output with designated suffix
save_path = os.path.join(paths['save_dir'], prefix + '_annotated.csv')
# annotate some random tracks
annotate(arr, tracks, df, save_path)

# '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_Position.csv'
# '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr'
# '/Users/amcg0011/Data/pia-tracking/rand' # need to add makedir 