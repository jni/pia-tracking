import napari
import pandas as pd 


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
                 out_col='correct'):
        '''
        View and annotate individual tracks

        Parameters
        '''
        # Data
        # ----
        self.array = hypercubes
        self.tracks = tracks_list
        df[out_col] = [None for _ in range(len(df))]
        df['idx'] = [i for i in range(len(df))]
        df = df.set_index('idx')
        self.df = df
        self.im_scale = im_scale
        
        # Column names
        # ------------
        self.id_col = id_col
        self.time_col = time_col
        self.out_col = out_col

        # Current track
        # -------------
        self.viewer = None
        self.i = 0
        ID = df[self.id_col][0]
        t = df[self.time_col][0]
        self.current = (ID, t)

        # Save
        # ----
        self.save_path = save_path


    # this method executes all other methods
    def annotate(self):
        with napari.gui_qt():
            self.viewer = napari.Viewer()
            self._show_track(self.viewer)
            print(f'Showing track 1 of {len(self.df)}')
            #self.viewer.unselect_all()
            self.viewer.bind_key('y', self.yes)
            self.viewer.bind_key('n', self.no)
            self.viewer.bind_key('d', self.next)
            self.viewer.bind_key('a', self.previous)


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
        self._remove_track(viewer)
        self._next(viewer)
        print(f'next')

    def previous(self, viewer):
        """
        Move to previous pair. Will be bound to key 'd'
        """
        self._remove_track(viewer)
        self._previous(viewer)
        print(f'previous')
    
    
    # Utility functions
    # -----------------

    def _annotate(self, ann, viewer):
        word = lambda a : 'correct' if a == 1 else 'incorrect'
        self._remove_track(viewer)
        self.df.at[self.i, self.out_col] = ann
        print(f'Track {self.current[0]} at time {self.current[1]} was marked {word(ann)}')
        self.df.to_csv(self.save_path) 
        self._next(viewer)
    
    
    def _next(self, viewer):
        if self.i + 1 < len(self.df):
            self.i = self.i + 1
            ID = self.df[self.id_col][self.i]
            t = self.df[self.time_col][self.i]
            self.current = (ID, t)
            self._remove_track(viewer)
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
            self._remove_track(viewer)
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


    def _remove_track(self, viewer):
        for _ in range(len(self.viewer.layers)):
            viewer.layers.pop(0)

    def _check_ann_status(self):
        out = self.df[self.out_col].values
        if None not in out:
            print('All tracks have been annotated')
        elif None in out and self.i + 1 == len(out):
            not_done = []
            for i, o in enumerate(out):
                if o == None:
                    not_done.append(i)
            if len(not_done) == 1:
                print(f'Track {not_done[0]} of {len(out)} has not yet been annotated')
            if len(not_done) > 1:
                print(f'Tracks {not_done} of {len(out)} have not yet been annotated')
        


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