from ast import literal_eval
from datetime import date
import dask.array as da
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from annotate_sample import annotate_track_terminations, annotate_tracks, annotate_object_tracks


def evaluate_tracks(tracks, 
                    image, 
                    save_path,
                    labels=None,
                    n_samples=30,
                    frames=30, 
                    box=60, 
                    id_col='ID',
                    time_col='t', 
                    array_order=('t', 'x' ,'y', 'z'),
                    scale=(1,1,1,4),
                    non_tzyx_col=None, 
                    seed=None,
                    weights=None, 
                    max_lost_prop=1.0,
                    min_track_length=20, # pick a number under which tracks are likely to be spurious
                    false_pos_col='FP', 
                    false_neg_col='FN',
                    **kwargs
                    ):
    """
    Three stage process (stages 1 and 2 complete)
        1) incorrect assignments - false positive
        2) incorrect termination -  false negative 
        3) failure to link objects - false negative 

    Parameters
    ----------
    tracks: pd.DataFrame 
    image: array-like
    save_path: str
    labels: None or array-like (int),
    n_samples: int,
    frames: int, 
    box: int, 
    id_col: str,
    time_col: str, 
    array_order: tuple of str
        E.g., ('t', 'x' ,'y', 'z') or ('c', 't', 'x' ,'y', 'z'). 
        Should correspond to names of columns in track data.
    scale: tuple of int,
    non_tzyx_col: str or tuple of str, 
    seed: None or int,
    weights: None or str, 
    max_lost_prop: float,
    min_track_length: int,
        Name a bit misleading. Refers to the min number of frames in which a
        track appears. 
    false_pos_col: str,
        Name of the column in which the instances of false positives will be 
        stored in annotation output csv. 
    false_neg_col:
        Name of the column in which the instances of false negatives will be 
        stored in annotation output csv.
    """
    os.makedirs(save_path, exist_ok=True)
    save_FP = get_new_path(save_path, '_FP.csv', 0)
    print('---------------------------------------------------------')
    info_FP = annotate_tracks(
                              tracks, 
                              image, 
                              save_FP,
                              labels=labels,
                              n_samples=n_samples,
                              frames=frames, 
                              box=box, 
                              id_col=id_col,
                              time_col=time_col, 
                              array_order=array_order,
                              scale=scale,
                              non_tzyx_col=non_tzyx_col, 
                              seed=seed,
                              weights=weights, 
                              max_lost_prop=max_lost_prop,
                              min_track_length=min_track_length,
                              false_pos_col=false_pos_col, 
                              false_neg_col=false_neg_col,
                              **kwargs
                              )
    save_TT = get_new_path(save_path, '_TT.csv', 0)
    print('---------------------------------------------------------')
    info_TT = annotate_track_terminations(tracks, 
                                          image, 
                                          save_TT,
                                          labels=labels,
                                          n_samples=n_samples,
                                          frames=frames, 
                                          box=box, 
                                          id_col=id_col,
                                          time_col=time_col, 
                                          array_order=array_order,
                                          scale=scale,
                                          non_tzyx_col=non_tzyx_col, 
                                          seed=seed,
                                          weights=weights, 
                                          max_lost_prop=max_lost_prop,
                                          min_track_length=min_track_length,
                                          false_pos_col=false_pos_col, 
                                          false_neg_col=false_neg_col,
                                          **kwargs)
    save_FN = get_new_path(save_path, '_FN.csv', 0)
    print('---------------------------------------------------------')
    hlf_frames = 16
    info_FN = annotate_object_tracks(tracks, 
                                     image, 
                                     save_FN,
                                     labels,
                                     n_samples=n_samples,
                                     frames=hlf_frames, 
                                     box=box, 
                                     id_col=id_col,
                                     time_col=time_col, 
                                     array_order=array_order,
                                     scale=scale,
                                     non_tzyx_col=non_tzyx_col, 
                                     seed=seed,
                                     weights=weights, 
                                     max_lost_prop=max_lost_prop,
                                     min_track_length=min_track_length,
                                     false_pos_col=false_pos_col, 
                                     false_neg_col=false_neg_col,
                                     **kwargs)
    dfs = [info_FP, info_TT, info_FN]
    for i, df in enumerate(dfs):
        for col in [false_pos_col, false_neg_col]:
            df = add_count_data(df, col, frames_col='frames')
            dfs[i] = df
    summary_df = generate_summary_df(dfs)
    sample = ['FP', 'TT', 'FN']
    summary_df.loc[:, 'Sample'] = sample
    summary_df = summary_df.set_index('Sample')
    save_summary = get_new_path(save_path, '_summary.csv', 0)
    summary_df.to_csv(save_summary)
    print('---------------------------------------------------------')
    print(f'Summary saved at {save_summary}')
    return dfs


def read_positives_data(data_dir):
    # LOL, why not glob? 
    annotations = [f for f in os.listdir(data_dir) if f.endswith('_annotation.csv')]
    df = pd.concat([pd.read_csv(os.path.join(data_dir, a)) for a in annotations])
    return df


# Helpers
# -------

def get_new_path(save_path, suffix, n):
    d = date.today().strftime('%y%m%d')
    p = Path(save_path)
    nm =  d + '_' + p.stem + '_' + str(n) + suffix
    new = os.path.join(save_path, nm)
    files = os.listdir(save_path)
    v = [True for f in files if f.find(nm) != -1]
    if True in v:
        n +=1
        return get_new_path(save_path, suffix, n)
    elif True not in v:
        return new


def add_count_data(df, col, frames_col='frames'): 
    # because frames is what the col t-min - t-max col is called
    if isinstance(df.loc[0, col], str):
        # this and the following isinstance lines are a tad fragile
        # will address this at a later date
        lens_p = [len(list(set(literal_eval(l)))) \
                  for l in df[col].values]
    else:
        m = 'Incorrect type for counting false positives'
        assert isinstance(df.loc[0, col], list), m
        lens_p = [len(list(set(l))) \
                  for l in df[col].values]
    df[col + '_count'] = lens_p
    df[col + '_per_frame'] = df[col + '_count'] / df[frames_col]
    return df


def generate_summary_df(dfs):
    col_names = [df.columns.values for df in dfs]
    #summary_cols = np.concatenate(col_names)
    means = [df.mean() for df in dfs]
    summary_cols = np.unique(np.concatenate([m.index.values for m in means]))
    sems = [df.sem() for df in dfs]
    rows = range(len(means))
    summary_df = {}
    for col in summary_cols:
        m_values = []
        s_values = []
        for i in range(len(rows)):
            if col in col_names[i]:
                new_m = means[i][col]
                new_s = sems[i][col]
            else:
                new_m, new_s = np.NaN, np.NaN
            m_values.append(new_m)
            s_values.append(new_s)
        summary_df[col + '_mean'] = m_values
        summary_df[col + '_sem'] = s_values
    summary_df = pd.DataFrame(summary_df)
    return summary_df


if __name__ == "__main__":
    from _parser import custom_parser, get_paths, track_view_base
    from data_io import single_zarr
    #
    # argparse parse arguments 
    #    - image data + tracks data + output directory 
    parser = custom_parser(image=True, tracks=True, save=True, labels=True)
    args = parser.parse_args()
    paths = get_paths(
                      args, 
                      'random_tracks_engine',
                      get={'data_path':'image', 
                           'tracks_path':'tracks',
                           'save_dir':'save', 
                           'labels_path': 'labels'
                           }, 
                      by_name=True
                      )
    # get random tracks from tracks and volume
    tracks_path = paths['tracks_path']
    tracks = pd.read_csv(tracks_path)
    image_path = paths['data_path']
    image = single_zarr(image_path)
    out_dir = paths['save_dir']
    prefix = Path(image_path).stem + '_' + Path(tracks_path).stem
    save_path = os.path.join(out_dir, prefix)
    labels_path = paths['labels_path']
    labels = da.from_zarr(labels_path)
    print(f'Output will be saved at {save_path}')
    dfs = evaluate_tracks(tracks, 
                          image, 
                          save_path, 
                          n_samples=30, 
                          labels=labels)




    #evaluate_positives(tracks_path, 
                     #  image_path, 
                     #  out_dir,
                     #  prefix, 
                     #  n_frames=[70,],
                     #  box_sizes=[80,], 
                     #  sample_size=3, 
                     #  id_col='ID', 
                     #  time_col='t', 
                     #  array_order=('t', 'x', 'y', 'z'), 
                     #  scale=(1, 1, 1, 4), 
                     #  )
     
    #data_dir = os.path.join(out_dir, prefix)
    # df = read_positives_data(data_dir)
    # positives_descriptives(df, data_dir, prefix)

    # -i /Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr -t /Users/amcg0011/GitRepos/pia-tracking/20200918-130313/btrack-tracks.csv -s /Users/amcg0011/Data/pia-tracking/
    # -i /Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr -t /Users/amcg0011/GitRepos/pia-tracking/20200918-130313/tracks.csv -s /Users/amcg0011/Data/pia-tracking/
    # /Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_btrack-tracks
    # /Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_tracks
    # OLD
    # ---
    #arr, tracks, df = get_random_tracks(paths, prefix, n=15)
    # save annotation output with designated suffix
    #save_path = os.path.join(paths['save_dir'], prefix + '_annotated.csv')
    # annotate some random tracks
    #annotate(arr, tracks, df, save_path)

    # '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_Position.csv'
    # '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr'
    # '/Users/amcg0011/Data/pia-tracking/rand' # need to add makedir \

    # python evaluate_tracks.py \
    # -i /Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3.zarr \
    # -t /Users/amcg0011/GitRepos/pia-tracking/20200918-130313/btrack-tracks.csv \
    # -s /Users/amcg0011/Data/pia-tracking/

