from annotate_tracks import annotate
import os
from random_tracks_engine import get_random_tracks, read_data, base
from parser import custom_parser, get_paths


if __name__ == "__main__":
    # parser
    parser = custom_parser(tracks=True, save=True, base=base)
    args = parser.parse_args()
    paths = get_paths(args, 
                      'random_tracks_engine',
                      get={'data_path':'image', 
                           'tracks_path':'track',
                           'save_dir':'save'}, 
                      by_name=True
                      )
    prefix = 'rand_tracks_3'
    arr, tracks, df = get_random_tracks(paths, prefix)
    save_path = os.path.join(paths['save_dir'], prefix + '_annotated.csv')
    annotate(arr, tracks, df, save_path)