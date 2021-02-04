from data_io import dict_2_JSON_serializable
from datetime import date
from itertools import repeat
import json
import numpy as np
from numpy import ndarray
import os
import pandas as pd 
from pandas import DataFrame
from pathlib import Path
from skimage.measure import regionprops
import time
from toolz import curry
from typing import Iterable, Union
from view_tracks import get_tracks, add_track_length


# ------------------------------
# RANDOM SAMPLE & BOUNDING BOXES
# ------------------------------

def random_sample(
                  df: DataFrame,
                  array: ndarray, 
                  name: str,
                  n_samples: int,
                  frames: int =10, 
                  box: int =60,
                  id_col: str ='ID', 
                  time_col: str ='t', 
                  array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                  scale: Iterable[int] =(1, 1, 1, 4), 
                  non_tzyx_col: Union[Iterable[str], str, None] = None,
                  seed: Union[int, None] =None,
                  weights: Union[str, None] =None,
                  max_lost_prop: Union[float, None] =None,
                  **kwargs
                  ):
    """
    Take a random sample from a 
    """
    shape = array.shape
    _frames = np.floor_divide(frames, 2)
    _box = np.floor_divide(box, 2)
    # this is the image shape scaled to the data
    scaled_shape = [s * scale[i] # scaled shape in image order
                    for i, s in enumerate(shape)]
    # curried function. Add important info for later
    _add_info = _add_sample_info(
                                 id_col, 
                                 time_col,
                                 _frames, 
                                 max_lost_prop)
    if seed is None:
        seed = np.random.randint(0, 100_000)
    # initalise the sample dict 
    #   Why? The function is recursive (if max_lost_prop < 1.0)
    #       initialising within isnt an option.
    #       and mutable defaults are just silly
    sample = {}
    sample = _sample(
                     sample, 
                     df, 
                     n_samples, 
                     seed, 
                     array_order, 
                     scaled_shape, 
                     name, 
                     weights, 
                     _add_info
                     )
    # sample scale is the scale that brings the data to the image
    sample = _estimate_bounding_boxes(
                                      sample, 
                                      shape,
                                      id_col, 
                                      time_col, 
                                      _frames, 
                                      _box,
                                      array_order, 
                                      non_tzyx_col, 
                                      scale
                                      )
    sample['info'] = _tracks_df(df, sample, id_col, time_col, array_order)
    return sample



# Sampling
# --------

def _sample(
            sample, 
            df, 
            n_samples, 
            seed, 
            array_order, 
            scaled_shape, 
            name, 
            weights, 
            _add_info):
    """
    select random sample
    """
    for i, col in enumerate(array_order):
        df = df.loc[(df[col] < scaled_shape[i])].copy()
    if n_samples == 0:
        print(f"Random {name} obtained")
        pairs = list(sample.keys())
        return sample
    elif n_samples < 0:
        # remove excess - print so that I can see if this is ever executed
        #   I don't think it will be, but just in case
        excess = abs(n_samples)
        print(f"Removing {excess} excess {name}")
        pairs = list(sample.keys())
        for i in range(excess):
            del sample[pairs[i]]
        return sample
    # If we don't get enough samples 
    #   (e.g., chooses a time point too near start or end)
    #   the function will call itself (with remainder samples to obtain)
    # NOTE: the max proportion of frames that can be lost in the track can be set 
    else:
        print(f'Sampling {n_samples}...')
        if weights is not None:
            w = df[weights].values
        else:
            w = None
        #iamkitten
        kittens_sample = df.sample(n=n_samples, weights=w, random_state=seed) # kitten's
        num_obtained = _add_info(kittens_sample, df, sample)
        n_samples = n_samples - num_obtained
        # this whole recursion thing doesnt really work when you use the same seed
        seed += 1 
        return _sample(
                       sample, 
                       df, 
                       n_samples, 
                       seed, 
                       array_order, 
                       scaled_shape, 
                       name, 
                       weights, 
                       _add_info)


# referenced in random_sample(...)
@curry
def _add_sample_info(id_col, 
                     time_col,
                     _frames, 
                     max_lost_prop,
                     sample_df, 
                     df, 
                     sample
                     ):
    counter = 0
    for i in range(len(sample_df)):
        ID = sample_df[id_col].values[i]
        t = sample_df[time_col].values[i]
        pair = (ID, t)
        lil_df = df.loc[
            (df[id_col] == ID) & 
            (df[time_col] >= t - _frames) & 
            (df[time_col] <= t + _frames)
            ]
        right_len = len(lil_df) >= np.floor((1-max_lost_prop) * (_frames * 2 + 1))
        right_len = right_len and len(lil_df) <= _frames * 2 + 1
        new_pair = pair not in sample.keys()
        row = df.loc[(df[id_col]==pair[0]) & 
                     (df[time_col]==pair[1])].copy()
        right_row_len = len(row) == 1
        if right_len & new_pair & right_row_len:
            lil_df = lil_df.reset_index(drop=True)
            try:
                lil_df = lil_df.drop(columns=['Unnamed: 0'])
            except:
                pass
            sample[pair] = {'df': lil_df}
            counter += 1
    return counter


# Bounding Boxes
# --------------

# referenced in random_sample(...)
def _estimate_bounding_boxes(
                             sample, 
                             shape,
                             id_col, 
                             time_col, 
                             _frames, 
                             _box,
                             array_order, 
                             non_tzyx_col, 
                             image_scale
                             ):
    print('Finding bounding boxes...')     
    hc_shape, sample_scale = _box_shape(
                                        array_order, 
                                        time_col, 
                                        non_tzyx_col, 
                                        _frames, 
                                        _box, 
                                        shape, 
                                        image_scale
                                        ) 
    coords_cols = _coords_cols(array_order, non_tzyx_col, time_col)
    pairs = sample.keys()
    for pair in pairs:
        df = sample[pair]['df']
        row = df.loc[(df[id_col]==pair[0]) & 
                     (df[time_col]==pair[1])].copy()
        sample[pair]['corr'] = {}
        # initialise the bounding box info
        sample[pair]['b_box'] = {}
        # TIME SLICE
        # find last time frame index
        for i, col in enumerate(array_order):
            if col == time_col:
                t_lim = shape[i] - 1
        sample = _time_slice(
                             pair, 
                             sample, 
                             df, 
                             t_lim, 
                             time_col, 
                             _frames, 
                             hc_shape
                             )
        # SPATIAL COORDS SLICES
        sample = _coords_slices(
                                array_order, 
                                coords_cols, 
                                hc_shape, 
                                sample_scale, 
                                row, 
                                _box, 
                                image_scale, 
                                sample,
                                shape,
                                df, 
                                pair
                                )
        # NON SPATIAL OR TIME SLICES
        sample = _non_tzyx_slices(sample, pair, non_tzyx_col)
    return sample


# referenced in _estimate_bounding_boxes
def _box_shape(
               array_order, 
               time_col, 
               non_tzyx_col, 
               _frames, 
               _box, 
               shape, 
               image_scale
               ):
    # get the scale that brings the data to the image
    sample_scale = []
    for i, col in enumerate(array_order):
            s = np.divide(1, image_scale[i])
            sample_scale.append(s)
    # if necessary, change configuration of non_tzyx_col
    if not isinstance(non_tzyx_col, Iterable):
        non_tzyx_col = [non_tzyx_col]
    # get the hypercube shape
    hc_shape = []
    for i, col in enumerate(array_order):
        if col == time_col:
            scaled = np.multiply(_frames*2+1, sample_scale[i])
            hc_shape.append(scaled)
        elif col in non_tzyx_col:
            scaled = np.multiply(shape[i]*2+1, sample_scale[i])
            hc_shape.append()
        else:
            scaled = np.multiply(_box*2+1, sample_scale[i])
            hc_shape.append(scaled)
    hc_shape = np.floor(np.array(hc_shape)).astype(int)
    return hc_shape, sample_scale


# referenced in _estimate_bounding_boxes
def _coords_cols(array_order, non_tzyx_col, time_col):
    if isinstance(non_tzyx_col, Iterable):
        coords_cols = non_tzyx_col.copy().append(time_col)
    else:
        coords_cols = [col for col in array_order if col \
                    not in [time_col, non_tzyx_col]]
    return coords_cols


# referenced in _estimate_bounding_boxes
def _time_slice(
                pair, 
                sample, 
                df, 
                t_lim, 
                time_col, 
                _frames, 
                hc_shape
                ):
    # get the estimated time index
    t_min, t_max = pair[1] - _frames, pair[1] + _frames
    # correct the min time index
    if t_min < 0:
        t_min = 0
    # correct the max time index
    if t_max > t_lim:
        t_max = t_lim
    t_max += 1
    # correct the track data for the bounding box
    df = sample[pair]['df']
    df[time_col] = df[time_col] - t_min 
    sample[pair]['b_box'][time_col] = slice(t_min, t_max)
    sample[pair]['df'] = df
    return sample


# referenced in _estimate_bounding_boxes
def _coords_slices(
                   array_order, 
                   coords_cols, 
                   hc_shape, 
                   sample_scale, 
                   row, 
                   _box, 
                   image_scale, 
                   sample,
                   shape,
                   df, 
                   pair
                   ):
    df = sample[pair]['df']
    for i, coord in enumerate(array_order):
        if coord in coords_cols:
            sz = hc_shape[i]
            # scale the tracks value to fit the image
            sample_2_box_scale = sample_scale[i]
            value = np.multiply(row[coord].values[0], sample_2_box_scale)
            # scale the bounding box for the image coordinate
            box = np.multiply(_box, sample_2_box_scale)
            col_min = np.floor(value - box).astype(int)
            col_max = np.floor(value + box).astype(int)
            sz1 = col_max - col_min
            diff = sz1 - sz
            col_min, col_max = _correct_diff(diff, col_min, col_max)
            # slice corrections
            box_to_tracks_scale = image_scale[i]
            # get the max and min values for track vertices in box scale
            max_coord = np.multiply(df[coord].max(), sample_2_box_scale)
            min_coord = np.multiply(df[coord].min(), sample_2_box_scale)
            # correct box shape if the box is too small to capture the 
            # entire track segment
            if min_coord < col_min:
                col_min = np.floor(min_coord).astype(int)
            if max_coord > col_max:
                col_max = np.ceil(max_coord).astype(int)
            if col_min < 0:
                col_min = 0
            if col_max > shape[i]:
                col_max = shape[i]
            sample[pair]['b_box'][coord] = slice(col_min, col_max)
            # need to construct correctional slices for getting 
            # images that are smaller than the selected cube volume
            # into the cube. 
            sz1 = col_max - col_min
            diff = sz - sz1
            # the difference should not be negative 
            #m0 = 'The cube dimensions should not be '
            #m1 = 'greater than the image in any axis: '
            #m2 = f'{col_min}:{col_max} for coord {coord}'
            #assert diff >= 0, m0 + m1 + m2
            # correct the position data for the frame
            c_min = (col_min * box_to_tracks_scale)
            df[coord] = df[coord] - c_min 
    return sample


# referenced in _coords_slices
def _correct_diff(diff, col_min, col_max):
    odd = diff % 2
    if diff < 0:
        diff = abs(diff)
        a = -1
    else:
        a = 1
    adj = np.floor_divide(diff, 2)
    col_min = col_min - (adj * a)
    if odd:
        adj = adj + 1
        col_max = col_max - (adj * a)
    else:
        col_max = col_max - (adj * a)
    return col_min, col_max


# referenced in _estimate_bounding_boxes
def _non_tzyx_slices(sample, pair, non_tzyx_col):
    if non_tzyx_col is not None:
        for col in non_tzyx_col:
            sample[pair]['b_box'][col] = slice(None)
    return sample


# referenced in random_sample(...)
def _tracks_df(df, sample, id_col, time_col, array_order):
    info = []
    pairs = sample.keys()
    for pair in pairs:
        # get the row of info about the sampled segment
        row = df.loc[(df[id_col]==pair[0]) & 
                     (df[time_col]==pair[1])].copy()
        # add summary stats for the track
        for col in sample[pair]['df'].columns.values:
            if isinstance(col[0], str):
                pass
            else:
                mean = sample[pair]['df'][col].mean()
                sem = sample[pair]['df'][col].sem()
                mean_name = col + '_seg_mean'
                sem_name = col + '_seg_sem'
                row.loc[:, mean_name] = mean
                row.loc[:, sem_name] = sem
        # add bounding box information 
        for coord in array_order:
            s_ = sample[pair]['b_box'][coord]
            s_min = s_.start
            n_min = coord + '_start'
            s_max = s_.stop
            n_max = coord + '_stop'
            row.loc[:, n_min] = [s_min,]
            row.loc[:, n_max] = [s_max,]
        num_frames = sample[pair]['df'][time_col].max() \
                    - sample[pair]['df'][time_col].min()
        row.loc[:, 'frames'] = [num_frames,]
        info.append(row)
    info = pd.concat(info)
    info = info.reset_index(drop=True)
    info = info.drop(columns=['Unnamed: 0'])
    return info 


# -------------
# SAMPLE TRACKS
# -------------

def sample_tracks(df: DataFrame,
                  array: ndarray, 
                  name: str,
                  n_samples: int,
                  frames: int =10, 
                  box: int =60,
                  id_col: str ='ID', 
                  time_col: str ='t', 
                  array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                  scale: Iterable[int] =(1, 1, 1, 4), 
                  non_tzyx_col: Union[Iterable[str], str, None] = None,
                  seed: Union[int, None] =None,
                  weights: Union[str, None] =None,
                  max_lost_prop: Union[float, None] =None,
                  min_track_length: Union[int, None] =20,
                  **kwargs):
    # calculate weights if required
    coords_cols = _coords_cols(array_order, non_tzyx_col, time_col)
    # well this was lazy (see below)
    df = _add_disp_weights(df, coords_cols, id_col)
    # older code from elsewhere, decided it wasn't hurting anything
    if weights is not None:
        if weights not in df.columns.values.tolist():
            if weights == '2-norm':
                df = _add_disp_weights(df, coords_cols, id_col)
            else: 
                m = 'Please use a column in the data frame or 2-norm to add distances'
                raise(KeyError(m))
    # filter for min track length
    df = add_track_length(df, id_col, new_col='track_length') 
    if min_track_length is not None:
        df = df.loc[df['track_length'] >= min_track_length, :]
    # get the sample
    sample = random_sample(
                           df,
                           array, 
                           name,
                           n_samples,
                           frames, 
                           box,
                           id_col, 
                           time_col, 
                           array_order, 
                           scale, 
                           non_tzyx_col,
                           seed,
                           weights,
                           max_lost_prop,
                           **kwargs)
    # add track arrays ready for input to napari tracks layer 
    sample = _add_track_arrays(sample, id_col, time_col, coords_cols)
    return sample


def _add_disp_weights(df, coords_cols, id_col):
        """
        Get L2 norm of finite difference across x,y,z for each track point
        These will be used as weights for random track selection.
        """
        coords = coords_cols
        weights = []
        for ID in list(df[id_col].unique()):
            # get the finite difference for the position vars
            diff = df.loc[(df[id_col] == ID)][coords].diff()
            diff = diff.fillna(0)
            # generate L2 norms for the finite difference vectors  
            n2 = list(np.linalg.norm(diff.to_numpy(), 2, axis=1))
            weights.extend(n2)
        v0 = len(weights)
        v1 = len(df)
        m = 'An issue has occured when calculating track displacements'
        m = m + f': the length of the data frame ({v1}) does not equal '
        m = m + f'that of the displacements ({v0})'
        assert v0 == v1, m 
        df['2-norm'] = weights
        return df


def _add_track_arrays(sample, id_col, time_col, coords_cols):
    cols = [id_col, time_col]
    for c in coords_cols:
        cols.append(c)
    for pair in sample.keys():
        if isinstance(pair, tuple):
            df = sample[pair]['df']
            tracks = df[cols].to_numpy()
            sample[pair]['tracks'] = tracks
    return sample


# -------------------------
# SAMPLE TRACK TERMINATIONS
# -------------------------

def sample_track_terminations(df: DataFrame,
                              array: ndarray, 
                              name: str,
                              n_samples: int,
                              frames: int =10, 
                              box: int =60,
                              id_col: str ='ID', 
                              time_col: str ='t', 
                              array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                              scale: Iterable[int] =(1, 1, 1, 4), 
                              non_tzyx_col: Union[Iterable[str], str, None] = None,
                              seed: Union[int, None] =None,
                              weights: Union[str, None] =None,
                              max_lost_prop: Union[float, None] =None,
                              min_track_length: Union[int, None] =20,
                              **kwargs
                              ):
    # filter data frame for terminations
    #
    # get sample
    sample = sample_tracks(
                           df,
                           array, 
                           name,
                           n_samples,
                           frames, 
                           box,
                           id_col, 
                           time_col, 
                           array_order, 
                           scale, 
                           non_tzyx_col,
                           seed,
                           weights,
                           max_lost_prop,
                           min_track_length,
                           **kwargs
                           )
    return sample


# --------------
# SAMPLE OBJECTS
# --------------


def sample_objects_bad(
                  df: DataFrame,
                  array: ndarray, 
                  name: str,
                  n_samples: int,
                  tracks_df: Union[None, pd.DataFrame] =None,
                  frames: int =10, 
                  box: int =60,
                  id_col: str ='ID', 
                  time_col: str ='t', 
                  array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                  scale: Iterable[int] =(1, 1, 1, 4), 
                  non_tzyx_col: Union[Iterable[str], str, None] = None,
                  seed: Union[int, None] =None,
                  weights: Union[str, None] =None,
                  max_lost_prop: Union[float, None] =None,
                  **kwargs
                  ):
    # probably need to add some preprocessing (perhaps use labels array to produce df for this)
    sample = random_sample(
                           df,
                           array, 
                           name,
                           n_samples,
                           frames, 
                           box,
                           id_col, 
                           time_col, 
                           array_order, 
                           scale, 
                           non_tzyx_col,
                           seed,
                           weights,
                           max_lost_prop,
                           **kwargs)
    if tracks_df is not None:
        for key in sample:
            if isinstance(key, tuple):
                lil_tracks = tracks_df.copy() 
                for c in array_order:
                    if c != time_col:
                        start = sample[key]['b_box'][c].start
                        stop = sample[key]['b_box'][c].stop
                        lil_tracks = lil_tracks.loc[(lil_tracks[c] > start) & (lil_tracks[c] < stop)]
                    else:
                        start = sample[key]['b_box'][c].start - np.floor_divide(frames, 2)
                        stop = sample[key]['b_box'][c].start + np.floor_divide(frames, 2)
                sample[key]['df'] = lil_tracks
        coords_cols = _coords_cols(array_order, non_tzyx_col, time_col)
        sample = _add_track_arrays(sample, id_col, time_col, coords_cols)
    return sample


def sample_objects_old(
                   objects,
                   tracks,
                   labels,
                   array, 
                   name,
                   n_samples,
                   frames, 
                   box,
                   id_col, 
                   time_col, 
                   array_order, 
                   scale, 
                   non_tzyx_col,
                   seed,
                   max_lost_prop=1.0,
                   **kwargs
                   ):
    shape = array.shape
    _frames = np.floor_divide(frames, 2)
    _box = np.floor_divide(box, 2)
    # sample objects
    #iamkitten
    kittens_sample = objects.sample(n=n_samples, random_state=seed) # kitten's
    sample = _get_object_tracks(labels,
                       kittens_sample, 
                       tracks, 
                       shape,
                       _frames, 
                       _box, 
                       scale, 
                       id_col, 
                       time_col, 
                       array_order, 
                       non_tzyx_col
                       )
    return sample


def _get_object_tracks(labels,
                       df, 
                       tracks, 
                       shape,
                       _frames, 
                       _box, 
                       scale, 
                       id_col, 
                       time_col, 
                       array_order, 
                       non_tzyx_col
                       ):
    # some scaled values for later
    # this is the image shape scaled to the data
    scaled_shape = [np.round(s * scale[i]).astype(int) # scaled shape in image order
                    for i, s in enumerate(shape)]
    inv_scale = np.divide(1, scale)
    hlf_hc = []
    for c in array_order:
        if c == time_col:
            n = _frames
        else:
            n = _box
        hlf_hc.append(n)
    # generate sample dict
    sample = {}
    df = df.reset_index(drop=True)
    df['frames'] = [None] * len(df)
    df['n_objects'] = [None] * len(df)
    df['t_start'] = [None] * len(df)
    # go through the sample and compose bounding boxes
    for idx in df.index:
        point_coords = []
        # get pair info for each sampled object
        # used as key for each object
        pair = (df.loc[idx, id_col], df.loc[idx, time_col])
        sample[pair] = {}
        # generate df
        lil_tracks = tracks.copy()
        b_box = {}
        for i, c in enumerate(array_order):
            coord = df.loc[idx, c]
            max_ = np.round(coord * scale[i] + hlf_hc[i])
            min_ = np.round(coord * scale[i] - hlf_hc[i])
            # correct for edges of image
            if max_ >= scaled_shape[i]:
                max_ = scaled_shape[i]
            if min_ <  0:
                min_ = 0
            # get all tracks that live in this box 
            lil_tracks = lil_tracks.loc[(lil_tracks[c] >= min_) & (lil_tracks[c] < max_)]
            if c == time_col:
                df.loc[i, 't_start'] = min_
            lil_tracks[c] = lil_tracks[c] - min_ 
            # same coordinates in image slicing scale
            b_min = np.round(coord - hlf_hc[i] * inv_scale[i]).astype(int)
            b_max = np.round(coord + hlf_hc[i] * inv_scale[i]).astype(int)
            # correct for edges of image
            if b_max >= shape[i]:
                b_max = shape[i] - 1
            if b_min < 0:
                b_min = 0 
            # add the coordinate slice to the pair info 
            b_box[c] = slice(b_min, b_max)
            # add the coordinate in image scale to the point coords
            point_coords.append(coord - b_min)
        # add the point to the pair
        sample[pair]['point'] = np.array([point_coords])
        # add the hypercube slices to the paie
        sample[pair]['b_box'] = b_box
        # add the tracks info to the pair
        lil_tracks = lil_tracks.reset_index(drop=True)
        sample[pair]['df'] = lil_tracks
        # get the tracks array for input to napari
        cols = [id_col, time_col]
        coord_cols = _coords_cols(array_order, non_tzyx_col, time_col)
        for c in coord_cols:
            cols.append(c)
        only_tracks = lil_tracks[cols].to_numpy()
        sample[pair]['tracks'] = only_tracks
    # add the sample info to the sample dict
    sample['info'] = df
    return sample
        


def sample_objects(
                   tracks,
                   labels,
                   array, 
                   name,
                   n_samples,
                   frames, 
                   box,
                   id_col, 
                   time_col, 
                   array_order, 
                   scale, 
                   non_tzyx_col,
                   seed,
                   max_lost_prop,
                   **kwargs
                   ): 
    #
    _frames = np.floor_divide(frames, 2)
    _box = np.floor_divide(box, 2)
    objs = get_objects_without_tracks(
                               labels, 
                               tracks, 
                               id_col, 
                               time_col, 
                               array_order, 
                               scale,
                               _frames, 
                               _box
                               )
    #
    try:
        kittens_sample = objs.sample(n=n_samples, random_state=seed)
    except:
        kittens_sample = objs
        print(len(objs))
    #
    shape = labels.shape
    sample = _get_object_tracks(labels,
                       kittens_sample, 
                       tracks, 
                       shape,
                       _frames, 
                       _box, 
                       scale, 
                       id_col, 
                       time_col, 
                       array_order, 
                       non_tzyx_col
                       )
    return sample

            
def get_objects_without_tracks(
                               labels, 
                               tracks, 
                               id_col, 
                               time_col, 
                               array_order, 
                               scale,
                               _frames, 
                               _box
                               ):
    """
    The 
    """
    df = {c:[] for c in array_order}
    df[id_col] = []
    df['area'] = []
    coord_cols = [c for c in array_order if c != time_col]
    coord_scale = [scale[i] for i, c in enumerate(array_order) if c != time_col]
    for t in range(labels.shape[0] - 1):
        try:
            frame = np.array(labels[t, ...])
            # get the tracks at this point in time 
            t_trk = tracks.copy()
            t_trk = t_trk.loc[t_trk[time_col] == t]
            # get the properties of objects in the frame
            props = regionprops(frame)
            # go through properties. 
            no_tracks = []
            for p in props:
                label = p['label']
                bbox = p['bbox']
                ct = t_trk.copy()
                # get any tracks in the bounding box for this obj
                for i, c in enumerate(coord_cols):
                    min_ = bbox[i] * coord_scale[i]
                    max_ = bbox[i + len(coord_cols)] * coord_scale[i]
                    ct = ct.loc[(ct[c] >= min_) & (ct[c] < max_)]
                # if there are no tracks in the bbox, add to the list
                if len(ct) == 0:
                    no_tracks.append(label)
            # based on list entries, make dataframe for objects
            for p in props:
                if p['label'] in no_tracks:
                    df[time_col].append(t)
                    for i, c in enumerate(coord_cols):
                        df[c].append(p['centroid'][i])
                    df[id_col].append(p['label'])
                    df['area'].append(p['area'])
        except KeyError:
            print(t)
    df = pd.DataFrame(df)
    print(f'Found {len(df)} untracked objects')
    return df
        

            #trk_vert = np.zeros(frame.shape, dtype=bool)
            #s_ = [np.round(t_trk[c].values * inv_scale[i]).astype(int) for i, c in enumerate(array_order)]
            #s_ = np.array(s_)
            #trk_vert[s_] = True
            #trk_obj = np.where(trk_vert == True, frame, trk_vert)
            #IDs = np.unique(trk_obj).tolist()

 
        
        






         



# -----------
# SAVE SAMPLE 
# -----------

def save_sample(save_name, sample):
    """
    Parse sample info to JSON serialisable and save
    NEEDS DEBUGGING GAVE UP 
        - TE: int64 not seriablisable (even after dataframes corrected)
    """
    sample = sample.copy()
    d = date.today().strftime('%y%m%d')
    if not save_name.endswith('.sample'):
        save_name = save_name + '.sample'
    if os.path.exists(save_name):
        p = Path(save_name)
        #name = p.
    os.makedirs(save_name, exist_ok=True)
    index_path = os.path.join(save_name, 'file_index.json')
    info_path = os.path.join(save_name, 'info.csv')
    
    
    
    #JSONified = dict_2_JSON_serializable(sample)
    #new = {}
    #for key in JSONified.keys():
        #if isinstance(key, tuple):
         #   nk = str(key)
        #else:
        #    nk = key
        #new[nk] = JSONified[key]
   # print(new)
    #with open(save_path, 'w') as f:
     #   json.dump(new, f, indent=4)


def read_sample(sample_path):
    """
    Parse sample info from .sample 'file'
    """
    pass


def save_sample_array(save_path, sample, array):
    """
    Save image data corresponding to a sample
    """
    pass

