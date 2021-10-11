import napari
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sampling import sample_tracks, sample_track_terminations, sample_objects
from skimage.measure import regionprops
from typing import Union, Iterable



# ------------------
# SampleViewer CLASS
# ------------------


class SampleViewer:
    def __init__(self, 
                 sample, 
                 array_dict, 
                 save_path,
                 add_data_function,
                 mode='Track',
                 array_order=('t', 'x' ,'y', 'z'),
                 time_col='t',
                 id_col='ID',
                 false_pos_col='FP', 
                 false_neg_col='FN',
                 t_start='t_start'):
        '''
        View and annotate a sample

        Parameters
        ----------
        sample: dict
            Sample information dict the form 
            {'info' : pd.DataFrame, 
            (ID, t) : {
                       'df' : pd.DataFrame, 
                       'b_box': {
                                 '<coord name>' : slice, 
                                 ... 
                                 }
                       },
            ...
            }
        array_dict: dict
            dict of arrays to be displayed and layer types. 
            All will be sliced according to 'b_box' (above)
            Of the form:
            {'<name>' : {
                         'data' : <array like>,
                         'type' : <'image' or 'labels'>, 
                         'scale' : tuple
                          }, 
            ...
            }
        save_path: str
            Save the output from the annotation here. This 
            will be the sample's info dataframe with added
            columns: 'correct', '<false_pos_col>', <false_neg_col>'
        array_order: tuple of str
            reflects the order of dimensions. String labels should
            correspond to those used in sampling process and probably
            to 
        

        '''
        # Prepare the image and or label data
        names = [key for key in array_dict.keys()]
        array = [array_dict[name]['data'] for name in names]
        array_types = [array_dict[name]['type'] for name in names]
        scales = [array_dict[name]['scale'] for name in names]
        self.arrays = array
        self.names = names
        self.array_types = array_types
        self.scales = scales
        self.array_order = array_order
        self.time_col = time_col
        self.id_col = id_col

        # sample info
        self.sample = sample
        self.info = sample['info']
        self.false_pos_col = false_pos_col # add FPs
        self.false_neg_col = false_neg_col # add FNs
        self.t_start = t_start
        self._add_annotation_cols()
        self.save_path = save_path # where the output goes
        
        # get the list of slices that will be used to display samples 
        self.pairs = [key for key in sample.keys() if isinstance(key, tuple)]
        self.slices = self._get_slices()
        self._i = 0

        # initialise viewer attr
        self.v = None

        # Add data (track, points, etc)
        self.mode = mode
        self.add_data_function = add_data_function


    def _add_annotation_cols(self):
        rl_info = range(len(self.info))
        corr = [None for _ in rl_info]
        fp = [[] for _ in rl_info]
        fn = [[] for _ in rl_info]
        self.info['correct'] = corr
        self.info[self.false_pos_col] = fp
        self.info[self.false_neg_col] = fn



    def _get_slices(self):
        slices = []
        for pair in self.pairs:
            s = []
            for coord in self.array_order:
                s.append(self.sample[pair]['b_box'][coord])
            slices.append(tuple(s))
        return slices


    @property
    def i(self):
        return self._i


    @i.setter
    def i(self, i):
        if i < len(self.slices) or i > 0:
            self._i = i
        else:
            print('invalid sample index')


    def _add_data(self):
        """
        Function to add data. Must take viewer, sample dict, and pair
        """
        self.add_data_function(self)


    def annotate(self):
        with napari.gui_qt():
            self._show_sample()
            self.v.bind_key('y', self.yes)
            self.v.bind_key('n', self.no)
            self.v.bind_key('d', self.next)
            self.v.bind_key('a', self.previous)
            # on the frame after a swap assign the frame number
                # to the ID swap list
            self.v.bind_key('i', self.false_positive)
            self.v.bind_key('Shift-i', self.false_negative)


    def _show_sample(self):
        # initialise the viewer if it doesnt exist
        if self.v is None:
            self.v = napari.Viewer()
            print('---------------------------------------------------------')
            print(f"Showing sample 1/{len(self.slices)}")
            print('---------------------------------------------------------')
            m = 'Mark samples as correct or incorrect by pressing y or n, '
            print(m)
            m = 'repspectively'
            print(m)
            print('---------------------------------------------------------')
            m = 'Indicate a false positive or false negative has occured'
            print(m)
            m = 'by pressing i or Shift-i, respectively. Doing so will'
            print(m)
            m = 'record the frame index in which the error occured'
            print(m)
        # get the slices representing the current track
        s_ = self.slices[self._i]
        # get the names of layers currently attached to the viewer
        layer_names = [l.name for l in self.v.layers]
        prop = {}
        # iterate through the layers to be added
        for i, name in enumerate(self.names):
            if name not in layer_names:
                t = self.array_types[i]
                scale = self.scales[i]
                if t == 'image':
                    print(self.arrays[i][s_])
                    img = np.array(self.arrays[i][s_])
                    self.v.add_image(img, 
                                     name=name, 
                                     scale=scale, 
                                     colormap='bop orange')
                elif t == 'labels':
                    labs = np.array(self.arrays[i][s_])
                    print(labs.shape)
                    self.v.add_labels(labs, 
                                      name=name, 
                                      scale=scale)
                else:
                    raise ValueError(f'No support for layer type {t}')
            else:
                self.v.layers[name].data = self.arrays[i][s_]
        self._add_data()
        


    # For Key Binding
    #----------------

    def yes(self, viewer):
        """
        Annotate as correct. Will be bound to key 'y'
        """
        self._annotate(1)
        self._check_ann_status()
    
    def no(self, viewer):
        """
        Annotate as incorrect. Will be bound to key 'n'
        """
        self._annotate(0)
        self._check_ann_status()

    def next(self, viewer):
        """
        Move to next pair. Will be bound to key 'd'
        """
        print(f'next')
        self._next()

    def previous(self, viewer):
        """
        Move to previous pair. Will be bound to key 'a'
        """
        print(f'previous')
        self._previous()
    

    def false_positive(self, viewer):
        """
        Get the time frame at which ID swap happended
        """
        t = self._get_current()
        self.info[self.false_pos_col][self._i].append(t)
        print(f'False positive recorded at time {t}')
        if self.save_path is not None:
            self.info.to_csv(self.save_path)

    
    def false_negative(self, viewer):
        """
        Get the time frame at which ID swap happended
        """
        t = self._get_current()
        self.info[self.false_neg_col][self._i].append(t)
        print(f'False negative recorded at time {t}')
        if self.save_path is not None:
            self.info.to_csv(self.save_path)


    # Key binding helpers
    # -------------------

    def _next(self):
        penultimate = len(self.slices) - 2
        if self._i <= penultimate:
            self._i += 1
            self._show_sample()
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.slices)}")
        else:
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.slices)}")
            self._check_ann_status()
            print("To navagate to prior samples press the a key")

    
    def _previous(self):
        if self._i >= 1:
            self._i -= 1
            self._show_sample()
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.slices)}")
        else:
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.slices)}")
            self._check_ann_status()
            print("To navagate to the next sample press the d key")


    def _annotate(self, ann):
        word = lambda a : 'correct' if a == 1 else 'incorrect'
        self.info.at[self.i, 'correct'] = ann
        print(f'{self.mode} was marked {word(ann)}')
        self._get_score()
        if self.save_path is not None:
            self.info.to_csv(self.save_path) 
        self._next()


    def _get_current(self):
        current = self.v.dims.current_step[0]
        t0 = self.info[self.t_start][self._i]
        t = current + t0
        return t


    def _check_ann_status(self):
        out = self.info['correct'].values
        if None not in out:
            print('---------------------------------------------------------')
            print('All tracks have been annotated')
            print(f'Final score is {self.score * 100} %')
        elif None in out and self._i + 1 > len(out):
            not_done = []
            for i, o in enumerate(out):
                if o == None:
                    not_done.append(i)
            if len(not_done) == 1:
                print('---------------------------------------------------------')
                print(f'Track {not_done[0]} of {len(out)} has not yet been annotated')
            if len(not_done) > 1:
                print('---------------------------------------------------------')
                print(f'Tracks {not_done} of {len(out)} have not yet been annotated')


    def _get_score(self):
            """
            Get the proportion of correctly scored tracks
            """
            self.score = np.divide(self.info['correct'].sum(), len(self.info))


# ---------------------------
# ANNOTATE TRACKS - POSITIVES
# ---------------------------


def annotate_tracks(
                    df, 
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
    p = Path(save_path)
    save_path = os.path.join(p.parents[0], p.stem + '_annotations.csv')
    sample_save_path = os.path.join(p.parents[0], p.stem + '_sample.csv')
    sample = sample_tracks(
                           df, 
                           image, 
                           'Tracks', 
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
                           ** kwargs)
    # so that the sample can be revieved if need be
    # save_sample_info(sample_save_path, sample)
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print(sample)
    array_dict = {
        'Image' : {
                   'data' : image, 
                   'type' : 'image',
                   'scale' : scale
                   },
                  }
    if labels is not None:
        array_dict['Labels'] = {
                                'data' : labels, 
                                'type' : 'labels',
                                'scale' : scale
                                }
    sv = SampleViewer( 
                      sample, 
                      array_dict, 
                      save_path,
                      add_track_data,
                      mode='Track',
                      array_order=('t', 'x' ,'y', 'z'),
                      false_pos_col=false_pos_col, 
                      false_neg_col=false_neg_col,
                      t_start='t_start', # this will be the name of the column in
                      )                  # in which the time slice.start is stored
                                    # as is outputted by the sample_tracks fnc
    sv.annotate()
    return sv.info


def annotate_track_terminations(df, 
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
                                **kwargs):
    print('Obtaining terminating tracks')
    # t
    p = Path(save_path)
    # The line below is for use in evaluate_tracks.py. 
    # In general, this is a silly way to construct a name
    save_all = os.path.join(p.parents[0], p.stem[7:] + '_with-terminating.csv')
    if not os.path.exists(save_all):
        # find terminating tracks and add weights
        unique_ids = df[id_col].unique()
        # set weights for sampling to zero by default
            # pandas won't include zeros weights
        terminating = [0] * len(df)
        df['terminating'] = terminating
        counter = 0
        for ID in unique_ids:
            counter += 1
            id_df = df.loc[(df[id_col] == ID)]
            id_df = id_df.sort_values(time_col)
            row = id_df.iloc[[-1]]
            i = row.index[0]
            # assume each ID appears once at each timepoint
            ID, t = row.loc[i, id_col], row.loc[i, time_col]
            for idx, dim in enumerate(array_order):
                if dim == time_col:
                    t_idx = idx
            if t < image.shape[t_idx] - 1:
                i = df.loc[(df[id_col] == ID) & (df[time_col] == t)].index[0]
                # weight for sampling set to one
                df.loc[i, 'terminating'] = 1
        print(f'{counter} track terminations identified.')
        df.to_csv(save_all)
    else:
        df = pd.read_csv(save_all)
    save_path = os.path.join(p.parents[0], p.stem + '_terminations.csv')
    info = annotate_tracks(
                           df, 
                           image, 
                           save_path,
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
                           weights='terminating', 
                           max_lost_prop=max_lost_prop,
                           min_track_length=min_track_length, 
                           # pick a number under which tracks 
                           #      are likely to be spurious
                           false_pos_col=false_pos_col, 
                           false_neg_col=false_neg_col,
                           **kwargs
                           )
    return info



def add_track_data(sv):
    viewer = sv.v
    sample = sv.sample
    pair = sv.pairs[sv._i]
    layer_names = [l.name for l in viewer.layers]
    data = sample[pair]['tracks']
    prop = {'track_id': data[:, 0]}
    if 'Tracks' not in layer_names:
        viewer.add_tracks(
                          data, 
                          properties=prop,
                          color_by='track_id',
                          name='Tracks', 
                          colormap='viridis', 
                          tail_width=6, 
                          )
    else:
        #i = [i for i, name in enumerate(layer_names) if name == 'Tracks']
        #viewer.layers.pop(i[0])
        #viewer.add_tracks(
         #                 data, 
          #                properties=prop,
           #               color_by='track_id',
            #              name='Tracks', 
             #             colormap='viridis', 
              #            tail_width=6, 
               #           )
        viewer.layers.properties = prop
        viewer.layers.color_by = 'track_id'
        viewer.layers['Tracks'].data = data
        viewer.layers.color_by = 'track_id'






# ---------------------------
# ANNOTATE TRACKS - NEGATIVES
# ----------------------------

def annotate_object_tracks(tracks_df, 
                           image, 
                           save_path,
                           labels,
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
    Take a random sample of objects and assess associated false negatives
    and false positives assocaited with that object. 
    """
    p = Path(save_path)
    print('Obtaining detected object sample...')
    save_all = os.path.join(p.parents[0], p.stem[7:] + '_with-terminating.csv')
    t_idx = [i for i, c in enumerate(array_order) if c == time_col][0]
    if not os.path.exists(save_all):
        df = {
            'ID': [], 
            'area': [], 
            }
        for coord in array_order:
            df[coord] = []
        t_max = labels.shape[t_idx]
        s_ = [slice(None, None)]*len(labels.shape) 
        sans_t_coord = [c for c in array_order if c != time_col]
        # Enter the for loop from hell =P
        counter = 0
        for t in range(t_max):
            try:
                # get a single frame at a time
                s_[t_idx] = slice(t, t+1)
                s_tup = tuple(s_)
                labs = np.array(labels[s_tup])
                labs = np.squeeze(labs)
                props = regionprops(labs)
                # ID with some properties. Could correlate the 
                for i, obj in enumerate(props):
                    # dummy IDs because ID counts are restarted each frame
                    assert i + counter not in df['ID']
                    df['ID'].append(i + counter)
                    df['area'].append(obj['area'])
                    for i, coord in enumerate(sans_t_coord):
                        df[coord].append(obj['centroid'][i])
                    df[time_col].append(t)
                    counter += 1
            except:
                print(f'File not found for t == {t}')
        assert len(list(set(df['ID']))) == len(df['ID'])
        df = pd.DataFrame(df)
        df.to_csv(save_all)
    else:
        df = pd.read_csv(save_all)
    sample = sample_objects(#df,
                          tracks_df,
                          labels,
                          image, 
                          'Objects',
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
                          **kwargs)
    array_dict = {
        'Image' : {
                   'data' : image, 
                   'type' : 'image',
                   'scale' : scale
                   },
                  }
    array_dict['Labels'] = {
                            'data' : labels, 
                            'type' : 'labels',
                            'scale' : scale
                            }
    sv = SampleViewer( 
                      sample, 
                      array_dict, 
                      save_path,
                      add_tracks_and_object_mask,
                      mode='Track',
                      array_order=('t', 'x' ,'y', 'z'),
                      false_pos_col=false_pos_col, 
                      false_neg_col=false_neg_col,
                      t_start='t_start', # this will be the name of the column in
                      ) 
    sv.annotate()
    return sv.info


def add_tracks_and_object_mask(sv: SampleViewer):
    '''
    Add tracks and an additional labels layer with a
    mask for the sampled object
    '''
    add_track_data(sv)
    # compute and add mask
    viewer = sv.v
    layer_names = [l.name for l in viewer.layers]
    for i, name in enumerate(sv.names):
        if name == 'Labels':
            scale = sv.scales[i]
            labels = sv.arrays[i]
    assert labels is not None, 'No labels layer supplied'
    # mask for the object ID
    pair = sv.pairs[sv._i]
    labs = np.array(viewer.layers['Labels'].data)
    mask = labs == pair[0]
    if 'Object' not in layer_names:
        viewer.add_labels(mask, 
                          name='Object')
    else:
        viewer.layers['Object'].data = mask
    if sv.info.loc[sv._i, 'n_objects'] == None:
        labs = viewer.layers['Labels'].data
        n = 0
        for t in range(labs.shape[0]):
            l = np.array(labs[t])
            n += np.unique(l).size - 1
            # - 1 because 0 will be a unique entity
        sv.info.loc[sv._i, 'n_objects'] = n 
    if sv.info.loc[sv._i, 'frames'] == None:
        f = np.array(viewer.layers['Labels'].data).shape[0]
        sv.info.loc[sv._i, 'frames'] = f

