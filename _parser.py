import argparse
import os
from pathlib import Path
from typing import Dict, Union


def custom_parser(image: bool=True, 
               tracks: bool=False, 
               coords: bool=False,
               name: bool=True,
               save: bool=False, 
               labels: bool=False,
               base: dict={} ) -> argparse.ArgumentParser:
    """
    Get an argument parser with some common arguments and, if you like, 
    additional arguments.

    Parameters
    ----------
    image: bool
    tracks: bool
    name: bool
    base: dict

    Return
    ------
    parser: argparse.ArgumentParser
    """
    parser_info = parser_dict(image=image, 
                              tracks=tracks, 
                              name=name, 
                              coords=coords, 
                              save=save,
                              labels=labels,
                              base=base)
    parser = get_parser(parser_info)
    return parser


def get_parser(parsing_info: Dict[Union[str], dict]
               ) -> argparse.ArgumentParser:
    """
    Get argparse parser from a dictionary containing necessary information

    Returns
    -------
    parser: argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser()
    for key in parsing_info.keys():
        arg_info = parsing_info[key]
        arg = arg_info['name']
        del arg_info['name']
        if isinstance(arg, str):
            parser.add_argument(arg, **arg_info)
        if isinstance(arg, list):
            parser.add_argument(arg[0], arg[1], **arg_info)
    return parser


def parser_dict(image: bool=True, 
               tracks: bool=False, 
               coords: bool=False,
               name: bool=True, 
               save: bool=False,
               labels: bool=False,
               base: dict={} ) -> Dict[Union[str], dict]:
    """
    Produce a dictionary with parser information. Can be provided with a base
    dictionary 

    Returns
    -------
    parser_info: dict of str, dict key value pairs

    """
    parser_info = base
    if image:
        arg = 'image'
        name = ['-i', '--image']
        h = "Input path to image data"
        parser_info[arg] = {'name' : name, 
                            'help' : h}
    if tracks:
        arg = 'tracks'
        name = ['-t', '--tracks']
        h = "Input path to tracks data"
        parser_info[arg] = {'name': name, 
                            'help': h}
    if name:
        arg = 'name'
        name = ['-n', '--name']
        h = 'name of user for custom hardcoded path/s'
        parser_info[arg] = {'name': name, 
                            'help': h}
    if coords:
        arg = 'coords'
        name = ['-c', '--coords']
        h = "Input path to platelet coordinates data"
        parser_info[arg] = {'name': name, 
                            'help': h}

    if save:
        arg = 'save'
        name = ['-s','--save']
        h = 'Input path to which to save output'
        default = 'btrack_tracks.csv'
        parser_info[arg] = {'name': name, 
                            'help': h, 
                            'default': default}
    
    if labels:
        arg = 'labels'
        name = ['-l','--labels']
        h = 'Input a path at which to find labels file'
        parser_info[arg] = {'name': name, 
                            'help': h, 
                            'default': default}

    return parser_info


# Specific Base Options
# ---------------------
h0 = "Minimum frames in which particles must appear to be "
h1 = "visualised as tracks (default = 20)"
track_view_base = {
        'min_frames' : {
            'name' :'--min_frames',
            'help' : h0 + h1, 
            'type' : int, 
            'default' : 20
        }
}

# Hacky Hardcoded Paths
# ---------------------

shortcuts = {
    'Abi' : {
        'view_tracks' : {
            'data_path' : '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr',
            'tracks_path' : '/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/tracks.csv'
        }, 
        'view_segmentation_3D' : {
            'data_path' : '/Users/amcg0011/Data/pia-tracking'
        }, 
        'btrack_tracking' : {
            'save_path' : "/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/btrack-tracks.csv", 
            'data_path' : '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr',
            'coords_path' : '/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/tracks.csv'
        },
        'random_tracks_engine' : {
            'save_dir' : "/Users/amcg0011/GitRepos/pia-tracking/20200918-130313", 
            'data_path' : '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr',
            'tracks_path' : '/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/btrack-tracks.csv', 
            'coords_path' : '/Users/amcg0011/GitRepos/pia-tracking/20200918-130313/tracks.csv'
        }
        
    },
    'Juan' : {
        'view_tracks' : {
            'data_path' : '/Users/jni/Dropbox/share-files/200519_IVMTR69_Inj4_dmso_exp3.nd2',
            'tracks_path' : '/Users/jni/Dropbox/share-files/tracks.csv'
        },
        'view_segmentation_3D' : {
            'data_path' : '/Users/jni/Dropbox/share-files/'
        }
    }
}

__file__var = str

def hardcoded_paths(name: str, file_: __file__var, by_name=False, shortcuts=shortcuts
                    ) -> Union[dict, str]:
    """
    get the variables for 
    """
    if not by_name:
        file_path = os.path.realpath(file_)
        file_name = str(Path(file_path).stem)
        values = shortcuts[name].get(file_name)
    elif isinstance(file_, str):
        values = shortcuts[name].get(file_)
    #else:
        #values = {**shortcuts[name].get(f) for f in file_}
    return values


# Utilities
# ---------

def get_paths(
              args,
              file_, 
              get={'data_path' : 'image', 
                  'tracks_path' : 'tracks'}, 
              by_name=False
                  ):
    if args.name:
        paths = hardcoded_paths(args.name, file_, by_name=by_name)
    else:
        args_ = vars(args)
        paths = {
            key : args_[get[key]] for key in get.keys()
        }
    return paths

    