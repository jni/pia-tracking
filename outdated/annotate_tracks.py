import napari
import numpy as np 
import pandas as pd 
from random_tracks_engine import RandomTracksEngine
from view_tracks import get_tracks


# -----------------------------------------------------------------------------
# TrackViewer Class
# -----------------

class TrackViewer:
    def __init__(self, 
                 hypercubes, 
                 tracks_list, 
                 df, 
                 save_path,
                 im_scale=(1, 1, 1, 4),
                 id_col='ID', 
                 time_col='t', 
                 out_col='correct', 
                 swap_col='ID_swaps', 
                 t_start='t_start'):
        '''
        View and annotate individual tracks

        Parameters
        '''
        # Data
        # ----
        self.array = None
        self.tracks = None
        self.df = None

        # Current track
        # -------------
        self.viewer = None
        self.i = None
        self.current = None

        # Column names
        # ------------
        self.id_col = id_col
        self.time_col = time_col
        self.out_col = out_col
        self.swap_col = swap_col
        self.t_start = t_start

        # Assign Vals
        # -----------
        self._data = self._set_data(hypercubes, tracks_list, df)
        self.im_scale = im_scale

        # Save
        # ----
        self.save_path = save_path

    # Data
    # ----

    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, data):
        hypercubes, tracks_list, df = data
        self._set_data(hypercubes, tracks_list, df)


    def _set_data(self, hypercubes, tracks_list, df):
        if hypercubes is not None:
            self.array = hypercubes
        if tracks_list is not None:
            self.tracks = tracks_list
        if df is not None:
            df[self.out_col] = [None for _ in range(len(df))]
            df[self.swap_col] = [[] for _ in range(len(df))]
            df['idx'] = [i for i in range(len(df))]
            df = df.set_index('idx')
            self.df = df
        if df is not None:
            self._initialise_current()
        self._data = self.array, self.tracks, self.df
        return self._data

    def _initialise_current(self):
        self.i = 0
        ID = self.df[self.id_col][0]
        t = self.df[self.time_col][0]
        self.current = (ID, t)


    # this method executes all other methods
    def annotate(self):
        with napari.gui_qt():
            if self.viewer is None:
                self.viewer = napari.Viewer()
            if len(self.viewer.layers) != 0:
                self._clear_viewer(self.viewer)
            self._show_track(self.viewer)
            print(f'Showing track 1 of {len(self.df)}')
            #self.viewer.unselect_all()
            self.viewer.bind_key('y', self.yes)
            self.viewer.bind_key('n', self.no)
            self.viewer.bind_key('d', self.next)
            self.viewer.bind_key('a', self.previous)
            # on the frame after a swap assign the frame number
                # to the ID swap list
            self.viewer.bind_key('i', self.add_id_swap)


    # For Key Binding
    #----------------

    def yes(self, viewer):
        """
        Annotate as correct. Will be bound to key 'y'
        """
        self._annotate(1, viewer)
    
    def no(self, viewer):
        """
        Annotate as incorrect. Will be bound to key 'n'
        """
        self._annotate(0, viewer)

    def next(self, viewer):
        """
        Move to next pair. Will be bound to key 'd'
        """
        self._clear_viewer(viewer)
        self._next(viewer)
        print(f'next')

    def previous(self, viewer):
        """
        Move to previous pair. Will be bound to key 'd'
        """
        self._clear_viewer(viewer)
        self._previous(viewer)
        print(f'previous')
    

    def add_id_swap(self, viewer):
        """
        Get the time frame at which ID swap happended
        """
        current = viewer.dims.current_step[0]
        t0 = self.df[self.t_start][self.i]
        t = current + t0
        self.df[self.swap_col][self.i].append(t)
    

    # Utility functions
    # -----------------

    def _annotate(self, ann, viewer):
        word = lambda a : 'correct' if a == 1 else 'incorrect'
        self._clear_viewer(viewer)
        self.df.at[self.i, self.out_col] = ann
        print(f'Track {self.current[0]} at time {self.current[1]} was marked {word(ann)}')
        self._get_score()
        if self.save_path is not None:
            self.df.to_csv(self.save_path) 
        self._next(viewer)


    def _get_score(self):
            """
            Get the proportion of correctly scored tracks
            """
            self.score = np.divide(self.df[self.out_col].sum(), len(self.df))
    
    
    def _next(self, viewer):
        if self.i + 1 < len(self.df):
            self.i = self.i + 1
            ID = self.df[self.id_col][self.i]
            t = self.df[self.time_col][self.i]
            self.current = (ID, t)
            self._clear_viewer(viewer)
            self._show_track(viewer)
            print(f'Showing track {self.i+1} of {len(self.df)}')
            self._check_ann_status()
        else:
            print('Final track in series, press a to go backwards')
            self._check_ann_status()
 

    def _previous(self, viewer):
        if self.i - 1 >= 0:
            self.i = self.i - 1
            ID = self.df[self.id_col][self.i]
            t = self.df[self.time_col][self.i]
            self.current = (ID, t)
            self._clear_viewer(viewer)
            self._show_track(viewer)
        else:
            msg0 = 'First track in series, press d to move on '
            msg1 = 'or y/no to annotate and move on'
            print(msg0 + msg1)


    def _show_track(self, viewer):
        i = self.i
        ID, t = self.current
        name = f'ID-{ID}_t-{t}'
        # add volume
        viewer.add_image(self.array[i], name='v_' + name, 
                         scale=self.im_scale)
        # add tracks layers
        viewer.add_tracks(self.tracks[i], name='t_' + name)


    def _clear_viewer(self, viewer):
        for _ in range(len(self.viewer.layers)):
            viewer.layers.pop(0)


    def _check_ann_status(self):
        out = self.df[self.out_col].values
        if None not in out:
            print('All tracks have been annotated')
            print(f'Final score is {self.score * 100} %')
        elif None in out and self.i + 1 > len(out):
            not_done = []
            for i, o in enumerate(out):
                if o == None:
                    not_done.append(i)
            if len(not_done) == 1:
                print(f'Track {not_done[0]} of {len(out)} has not yet been annotated')
            if len(not_done) > 1:
                print(f'Tracks {not_done} of {len(out)} have not yet been annotated')

    
# -----------------------------------------------------------------------------
# CostEvaluation class
# --------------------

class CostEvaluation(TrackViewer):
    def __init__(self,
                 tracks, 
                 array, 
                 shape, 
                 im_scale=(1, 1, 1, 4),
                 id_col='ID', 
                 time_col='t', 
                 out_col='correct'):
        super().__init__(
                       hypercubes=None, 
                       tracks_list=None, 
                       df=None, 
                       save_path=None,
                       im_scale=im_scale,
                       id_col=id_col, 
                       time_col=time_col, 
                       out_col=out_col
                       )
        self.all_tracks = tracks
        self.image = array
        self.shape = shape
        self.RTE = None
        self.cost = None


    def annotate_all(self):
        tracks = get_tracks(self.all_tracks, id_col='parent', time_col='t', w_prop=False)
        with napari.gui_qt():
            self.viewer = napari.Viewer()
            self.viewer.add_image(self.image, scale=self.im_scale)
            self.viewer.add_tracks(tracks, colormap='viridis')
            self.viewer.bind_key('y', self._approve)
            self.viewer.bind_key('n', self._disapprove)


    # For Key Binding
    # ---------------

    def _approve(self, viewer):
        self._clear_viewer(viewer)
        self.viewer.bind_key('y', None)
        self.viewer.bind_key('n', None)
        self.RTE = RandomTracksEngine(self.all_tracks, 
                                      self.image, 
                                      self.shape)
        hcs, random_tracks = self.RTE.add_tracks(30)
        info = self.RTE.tracks_info
        self.data = hcs, random_tracks, info
        self.annotate()
        self.cost = 1 - self.score


    def _disapprove(self, viewer):
        self._clear_viewer(viewer)
        self.cost = 1
        self.viewer.bind_key('n', None)
        self.viewer.bind_key('y', None)
        


# -----------------------------------------------------------------------------
# Functions
# ---------

def annotate(
             hypercubes, 
             tracks_list, 
             df, 
             save_path,
             im_scale=(1, 1, 1, 4),
             id_col='ID',
             time_col='t', 
             out_col='correct'
             ):
    track_viewer = TrackViewer(
                               hypercubes, 
                               tracks_list, 
                               df, 
                               save_path,
                               im_scale=im_scale,
                               id_col=id_col, 
                               time_col=time_col, 
                               out_col=out_col)
    track_viewer.annotate()
