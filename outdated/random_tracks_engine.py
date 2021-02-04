import dask.array as da
from data_io import single_zarr
from itertools import repeat
import json
import napari
import numpy as np
import os
import pandas as pd 
from _parser import custom_parser, get_paths, track_view_base
import time
from view_tracks import get_tracks, add_track_length
import zarr


# -----------------------------------------------------------------------------
# RandomTracksEngine Class
# ------------------------
class RandomTracksEngine:

    def __init__(self, 
                 tracks,
                 array, 
                 frames=10, 
                 box=60,
                 id_col='ID', 
                 time_col='t', 
                 array_order=('t', 'x', 'y', 'z'), 
                 scale=(1, 1, 1, 4), 
                 seed=None,
                 weighted=False,
                 max_lost_prop=None,
                 min_frames=20,
                 **kwargs 
                ):
        '''
        Class for obtaining samples of random tracks 

        Parameters
        ----------
        tracks: pd.DataFrame
            data frame containing the track vertices
        array: dask array
            image from which to obtain track hypercubes
        frame_shape: tuple of int
            what is the shape of a single frame
        t_max: int
            .
        frames: int
            .
        box: int
            .
        id_col: str
            .
        time_col: str
            .
        coords_cols: tuple of str
            .
        array_order: dict of str : int pairs
            .
        scale: tuple of int or float
            .

        Properties
        ----------
        #TODO

        Methods
        -------
        add_tracks: (list of ndarray, ndarray)
            .
        save: None
            .

        Notes
        -----
        This will need finessing as I find out what is 
        necessary for tracks appraisal.
        '''
        shape = array.shape
        t_max = array.shape[0]

        # Data
        # ----
        tracks = add_track_length(tracks, id_col, new_col='track_length') 
        if min_frames is not None:
            tracks = tracks.loc[tracks['track_length'] >= min_frames, :]
        self.tracks = tracks # tracks data frame
        self.array = array # image array

        # Image shapes
        # ------------
        self.frame_shape = shape[1:] # in z, y, x (array frame)
        self.t_max = t_max # number of frames in array
        self.image_shape = shape
        scaled_shape = [s * scale[i] # scaled shape in image order
                        for i, s in enumerate(shape)]
        self.scaled_shape = scaled_shape
        self.image_scale = scale
        

        # Tracks Information
        # ------------------
        self.time_col = time_col
        coords_cols = []
        for col in array_order:
            if col != time_col:
                coords_cols.append(col)
        self.coords_cols = coords_cols
        self.array_order = array_order
        self.id_col = id_col
        self.time_col = time_col
        tracks_scale = []
        for i, col in enumerate(array_order):
            s = np.divide(1, scale[i])
            tracks_scale.append(s)
        self.tracks_scale = tracks_scale


        # For Hypervolume 
        # ---------------
        self._frames = np.floor_divide(frames, 2)
        self._box = np.floor_divide(box, 2)
        hc_shape = []
        for i, col in enumerate(array_order):
            if col == self.time_col:
                scaled = np.multiply(self._frames*2+1, self.tracks_scale[i])
                hc_shape.append(scaled)
            else:
                scaled = np.multiply(self._box*2+1, self.tracks_scale[i])
                hc_shape.append(scaled)
        hc_shape = np.floor(np.array(hc_shape)).astype(int)
        self.hc_shape = hc_shape

        
        # Random tracks data
        # ------------------
        self.random_tracks = {}
        self.hypercubes = None
        self.tracks_list = []
        self.tracks_info = None

        # add magnetude of displacement between time-ponts
        # for each track vertex
        self._add_disp_weights()

        # Random tracks finding
        # ---------------------
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 100_000)
        self.weighted = weighted
        if max_lost_prop is None:
            max_lost_prop = 1.0
        self.max_lost_prop = max_lost_prop


    # PROPERTIES
    # ==========
    # TODO add the hypercubes/trackslist/tracksinfo
    # TODO rename things so that I can have public properties
    #@property
    #def 


    # PUBLIC METHODS
    # ==============

    # Random Tracks
    # -------------
    def add_tracks(self, num_tracks=10):
        '''
        Add random tracks
        '''
        pairs = self._grab_tracks(num_tracks, self.seed)
        print(f'Total tracks: {len(pairs)}')
        array = self._grab_hypercubes(pairs)
        tracks_list = self._list_of_tracks(pairs)
        tracks_info = self._tracks_df(pairs)
        # now validate the data
        self._validate_array(pairs, array)
        self._validate_tracks_list(pairs, tracks_list)
        self._validate_tracks_info(pairs, tracks_info)
        # now add the data to attributes
        if self.hypercubes is None:
            self.hypercubes = array
        else:
            hc = np.concatenate([self.hypercubes, array])
            self.hypercubes = hc
        for t in tracks_list:
            self.tracks_list.append(t)
        if self.tracks_info is None:
            self.tracks_info = tracks_info
        else:
            ti = pd.concat([self.tracks_info, tracks_info])
            self.tracks_info = ti
        return array, tracks_list
    

    # Save Content
    # ------------
    def save(self, prefix, save_dir):
        # save the array
        save_path = os.path.join(save_dir, prefix + '.zarr')
        zarr.save(save_path, self.hypercubes)
        # save the tracks data
        save_path = os.path.join(save_dir, prefix + '.json')
        tracks_out = {i : t.tolist() for i, t in enumerate(self.tracks_list)}
        with open(save_path, 'w') as f:
            op = json.dumps(tracks_out, indent=4)
            f.write(op)
        # save tracks info
        save_path = os.path.join(save_dir, prefix + '.csv')
        self.tracks_info.to_csv(save_path)
        m0 = f'Hypercube array, tracks data, and tracks info for {prefix} '
        m1 = f'saved at {save_dir}'
        print(m0 + m1)


    # PRIVATE METHODS
    # ===============

    # Weights
    # -------
    def _add_disp_weights(self):
        """
        Get L2 norm of finite difference across x,y,z for each track point
        These will be used as weights for random track selection.
        """
        coords = self.coords_cols
        df = self.tracks
        weights = []
        for ID in list(df[self.id_col].unique()):
            # get the finite difference for the position vars
            diff = df.loc[(df[self.id_col] == ID)][coords].diff()
            # the first value will be NaN - replace with the second
            # so as not to disadvantage the first point in a track
            #start = diff.index[0]
            # if len(diff) >= 2: 
               # for coord in coords:
                     # assumes index is a range)
                #    diff.loc[start, coord] = diff.loc[start + 1, coord]
            #else: 
               # for coord in coords:
                 #   diff.loc[start, coord] = 0
            # lazy way
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
        self.tracks['2-norm'] = weights


    # Random Tracks
    # -------------
    def _grab_tracks(self, num_tracks, s):
        """
        select random tracks
        """
        df = self.tracks
        for i, col in enumerate(self.array_order):
            df = df.loc[(df[col] < self.scaled_shape[i])]
        if num_tracks == 0:
            print("Random tracks obtained")
            pairs = list(self.random_tracks.keys())
            return pairs
        elif num_tracks < 0:
            # remove excess - print so that I can see if this is ever executed
            #   I don't think it will be, but just in case
            excess = abs(num_tracks)
            print(f"Removing {excess} excess tracks")
            pairs = list(self.random_tracks.keys())
            for i in range(excess):
                del self.random_tracks[pairs[i]]
            return pairs
        # If we don't get enough samples 
        #   (e.g., chooses a time point too near start or end)
        #   the function will call itself (with remainder samples to obtain)
        # NOTE: the max proportion of frames that can be lost in the track can be set 
        else:
            print(f'Sampling {num_tracks}...')
            if self.weighted:
                w = df['2-norm'].values
            else:
                w = None
            sample = df.sample(n=num_tracks, weights=w, random_state=s)
            num_obtained = self._add_track_info(sample, df)
            num_tracks = num_tracks - num_obtained
            return self._grab_tracks(num_tracks, s=s+1)


    # referenced in self._grab_tracks()
    def _add_track_info(self, sample, tdf):
        counter = 0
        for i in range(len(sample)):
            ID = sample[self.id_col].values[i]
            t = sample[self.time_col].values[i]
            pair = (ID, t)
            df = tdf.loc[
                (tdf[self.id_col] == ID) & 
                (tdf[self.time_col] >= t - self._frames) & 
                (tdf[self.time_col] <= t + self._frames)
                ]
            right_len = len(df) >= np.floor((1-self.max_lost_prop) * (self._frames * 2 + 1))
            right_len = right_len and len(df) <= self._frames * 2 + 1
            new_pair = pair not in self.random_tracks.keys()
            row = tdf.loc[(tdf[self.id_col]==pair[0]) & 
                         (tdf[self.time_col]==pair[1])]
            right_row_len = len(row) == 1
            if right_len & new_pair & right_row_len:
                self.random_tracks[pair] = {'df': df}
                counter += 1
        return counter


    # Hypercubes
    # ----------
    # referenced in self.add_tracks()
    def _grab_hypercubes(self, pairs):
        self._estimate_bounding_boxes(pairs)
        array = self._get_array(pairs)
        return array


    # referenced in self._grab_hypercubes()
    def _estimate_bounding_boxes(self, pairs):
        print('Finding bounding boxes...')
        for pair in pairs:
            df = self.random_tracks[pair]['df']
            row = df.loc[(df[self.id_col]==pair[0]) & 
                         (df[self.time_col]==pair[1])]
            self.random_tracks[pair]['corr'] = {}
            df = self._time_slice(df, pair)
            df = self._coords_slices(row, df, pair)
            self.random_tracks[pair]['df'] = df


    # referenced in self._estimate_bounding_boxes()
    def _time_slice(self, df, pair):
        t_min, t_max = pair[1] - self._frames, pair[1] + self._frames
        df[self.time_col] = df[self.time_col] - t_min 
        self.random_tracks[pair]['b_box'] = {self.time_col : slice(t_min, 
                                                                       t_max,)}
        self.random_tracks[pair]['corr'][self.time_col] = slice(0, self.hc_shape[0])
        return df


    # referenced in self._estimate_bounding_boxes()
    def _coords_slices(self, row, df, pair):
        for i, coord in enumerate(self.array_order):
            if coord in self.coords_cols:
                sz = self.hc_shape[i]
                # scale the tracks value to fit the image
                tracks_2_box_scale = self.tracks_scale[i]
                value = np.multiply(row[coord].values[0], tracks_2_box_scale)
                # scale the bounding box for the image coordinate
                box = np.multiply(self._box, tracks_2_box_scale)
                col_min = np.floor(value - box).astype(int)
                col_max = np.floor(value + box).astype(int)
                sz1 = col_max - col_min
                diff = sz1 - sz
                col_min, col_max = self._correct_diff(diff, col_min, col_max)
                # slice corrections
                box_to_tracks_scale = self.image_scale[i]
                c_min = (col_min * box_to_tracks_scale)
                if col_min < 0:
                    col_min = 0
                    a = 1
                if col_max > self.image_shape[i]:
                    col_max = self.image_shape[i]
                    a = -1
                self.random_tracks[pair]['b_box'][coord] = slice(col_min, 
                                                                 col_max)
                # need to construct correctional slices for getting 
                # images that are smaller than the selected cube volume
                # into the cube. 
                sz1 = col_max - col_min
                diff = sz - sz1
                # the difference should not be negative 
                m0 = 'The cube dimensions should not be '
                m1 = 'greater than the image in any axis'
                assert diff >= 0, m0 + m1
                if diff > 0: # the difference should not be negative 
                    s_min = np.floor_divide(diff, 2)
                    c_tks = s_min * box_to_tracks_scale * a
                    if diff % 2:
                        s_max = sz - s_min - 1
                    else:
                        s_max = sz - s_min
                else:
                    s_min = 0
                    s_max = sz
                    c_tks = 0
                # add correction slice
                self.random_tracks[pair]['corr'][coord] = slice(s_min, s_max)
                # correct the tracks data to fit in the volume
                c_min = c_min + c_tks
                df[coord] = df[coord] - c_min #
        return df
            

    # referenced in self._coords_slices()
    def _correct_diff(self, diff, col_min, col_max):
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

    
    # referenced in self._grab_hypercubes()   
    def _get_array(self, pairs):
        print('Obtaining hypercubes...')
        t0 = time.time()
        arrays = []
        first = None
        first = self.array_order[0]
        assert first != None
        for pair in pairs:
            slices  = self.random_tracks[pair]['b_box']
            # get the slice for the first dim of array
            fs = slices[first]
            dif = self.hc_shape[0]
            sub = []
            for i in range(dif):
                idx = fs.start + i
                if idx >= self.t_max or idx < 0:
                    frame = np.zeros(self.frame_shape)
                else:
                    frame = self.array[idx, ...].compute()
                sub.append([frame])
            sub = np.concatenate(sub)
            s_ = [slices[coord] for coord in self.array_order]
            s_[0] = slice(0, dif)
            s_ = tuple(s_)
            sml_sub = sub[s_]
            sml_arr = self._correct_shape(sml_sub, pair)
            arrays.append([sml_arr])
        arrays = np.concatenate(arrays)
        t1 = time.time()
        print(f'Hypercube array of shape {arrays.shape} obtained in {t1-t0} seconds.')
        return arrays
        

    # referenced in self._get_arrays()
    def _correct_shape(self, sub, pair):
        corr = self.random_tracks[pair]['corr']
        slice_ = []
        for coord in self.array_order:
            slice_.append(corr[coord])
        slice_ = tuple(slice_)
        arr = np.zeros(self.hc_shape, dtype=sub.dtype)
        arr[slice_] = sub
        return arr


    # Tracks for Napari
    # -----------------
    def _list_of_tracks(self, pairs):
        print('Gathering track data...')
        track_list = []
        for pair in pairs:
            df = self.random_tracks[pair]['df']
            data_cols = [self.id_col, self.time_col] + list(self.coords_cols)
            track_data = df[data_cols].to_numpy()
            track_list.append(track_data)
            print(type(track_data), track_data.shape)
        print(f'Data obtained for {len(track_list)} tracks')
        print(type(track_list), len(track_list), type(track_list[0]))
        return track_list


    # Random Tracks DataFrame
    # -----------------------
    def _tracks_df(self, pairs):
        print('Gathering track volumes info...')
        info = []
        for pair in pairs:
            df = self.tracks
            row = df.loc[(df[self.id_col]==pair[0]) & 
                         (df[self.time_col]==pair[1])]
            mean_l2n = self.random_tracks[pair]['df']['2-norm'].mean()
            row['mean_2-norm'] = mean_l2n
            for coord in self.array_order:
                s_ = self.random_tracks[pair]['b_box'][coord]
                s_min = s_.start
                n_min = coord + '_start'
                s_max = s_.stop
                n_max = coord + '_stop'
                row.loc[:, n_min] = [s_min,]
                row.loc[:, n_max] = [s_max,]
            row.loc[:, 'frames'] = [self._frames * 2 + 1,]
            info.append(row)
        info = pd.concat(info)
        print(f'Track volume info obtained for {len(info)} tracks')
        return info    


    # Validate
    # --------
    def _validate_array(self, pairs, array):
        # check the number of track volumes
        v = array.shape[0]
        m = f"the number of cubes ({v}) doesn't match tracks ({len(pairs)})"
        assert v == len(pairs), m
        # check the dims of track volumes
        v0 = tuple(self.hc_shape)
        v1 = tuple(array.shape[1:])
        m = f"the cubes do not match the expected shape, {v1} vs {v0}"
        assert v0 == v1, m


    def _validate_tracks_list(self, pairs, tracks_list):
        v0 = len(tracks_list)
        v1 = len(pairs)
        m = f'the number of track arrays {v0} does not match that selected{v1}'
        assert v0 == v1, m


    def _validate_tracks_info(self, pairs, tracks_info):
        v0 = len(tracks_info)
        v1 = len(pairs)
        m0 = f'the number of track arrays {v0} '
        m1 = f'does not match those in info data frame {v1}'
        assert v0 == v1, m0 + m1


# -----------------------------------------------------------------------------
# Functions
# ---------

# The Wrapper 
# -----------
def get_random_tracks(
                      paths, 
                      prefix, 
                      n=10, 
                      t_max=193, 
                      frames=10, 
                      box=60,
                      id_col='ID', 
                      time_col='t', 
                      array_order=('t', 'x', 'y', 'z'), 
                      scale=(1, 1, 1, 4), 
                      seed=None, 
                      weighted=False, 
                      max_lost_prop=0.1
                      ):
    df, array = read_data(paths)
    RTE = RandomTracksEngine(
                             df, 
                             array, 
                             frames=frames, 
                             box=box,
                             id_col=id_col, 
                             time_col=time_col, 
                             array_order=array_order, 
                             scale=scale, 
                             seed=seed, 
                             weighted=weighted, 
                             max_lost_prop=max_lost_prop
                             )
    hcs, tracks = RTE.add_tracks(n)
    RTE.save(prefix, paths['save_dir'])
    return hcs, tracks, RTE.tracks_info


# Data IO
# -------
def read_data(paths):
    df = pd.read_csv(paths['tracks_path'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    #arr, shape = get_stack(paths['data_path'], t_max=t_max, w_shape=True)
    arr = single_zarr(paths['data_path'])
    return df, arr


# View Result
# -----------
def view_random_tracks(tracks, array):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(array, scale=(1, 1, 1, 1, 4))
        for i, tk in enumerate(tracks):
            name = f"track_{i}"
            viewer.add_tracks(tk, name=name, visible=False)


# -----------------------------------------------------------------------------
# Execution
# ---------
if __name__ == "__main__":
    # parser
    parser = custom_parser(tracks=True, save=True, base=track_view_base)
    args = parser.parse_args()
    paths = get_paths(args, 
                      __file__,
                      get={'data_path':'image', 
                           'tracks_path':'track',
                           'save_dir':'save' 
                           })
    # random tracks
    df, array = read_data(paths)
    prefix = 'rand_tracks_0'
    RTE = RandomTracksEngine(df, array)
    hcs, tracks = RTE.add_tracks(10)
    RTE.save(prefix, paths['save_dir'])
    view_random_tracks(tracks, hcs)

