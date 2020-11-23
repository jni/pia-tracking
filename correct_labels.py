from data_io import single_zarr
import os
import napari
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import octahedron
from skimage.segmentation import watershed
import zarr


def correct_labels(image_path, labels_path, frame, scale=(1,1,4), c=2):
    """
    Correct labels to create a ground truth with four opperations:
        (1) points - to seed watershed to split labels.
        (2) watershed - recompute watershed for joined labels based on 
            chosen point
        (3) pick label colour - choose the colour of the label with which
            to paint.
        (4) paint - paint new label. 
    """
    lets_annotate = CorrectLabels(image_path, labels_path, frame, scale=scale, c=c)
    lets_annotate()



class CorrectLabels:
    def __init__(self, image_path, labels_path, frame, scale=(1,1,4), c=2):
        """
        Correct labels to create a ground truth with five opperations, each of which 
        correspond to the following number key. 
            (1) toggel selection of points - to seed watershed to split labels.
            (2) watershed - recompute watershed for joined labels based on 
                chosen points.
            (3) toggle pick label colour - choose the colour of the label with which
                to paint.
            (4) toggle label fill mode - for merging labels
            (5) toggel paint mode - paint new label. 
        Output can be saved by using Control-s 
        """
        self.scale = scale
        # Read in data
        self.array = single_zarr(image_path, c=c)[frame].compute()
        if labels_path.endswith('_GT.zarr'):
            self.labels = np.array(zarr.open(labels_path, mode='r+'))
            # save path is the labels path
            self.save_path = labels_path
        else:
            self.labels = zarr.open(labels_path, mode='r+')[frame]
            # Get the save path 
            data_path = Path(labels_path)
            self.save_path = os.path.join(data_path.parents[0], 
                                          data_path.stem + f'_t{frame}_GT.zarr')


    def __call__(self):
        with napari.gui_qt():
            v = napari.Viewer()
            print(type(self.labels))
            v.add_image(self.array, name='Image', scale=self.scale)
            v.add_labels(self.labels, name='Labels', scale=self.scale)
            v.add_points(np.empty((0, len(self.labels.shape)), dtype=float), scale=self.scale, size=2)
            v.bind_key('1', self._points)
            v.bind_key('2', self._watershed)
            v.bind_key('3', self._select_colour)
            v.bind_key('4', self._fill)
            v.bind_key('5', self._paint)
            v.bind_key('Shift-s', self._save)
    

    def _points(self, viewer):
        """
        Switch to points layer to split a label
        """
        if viewer.layers['Points'].mode != 'add':  
            viewer.layers['Points'].mode = 'add' 
        else:   
            viewer.layers['Points'].mode = 'pan_zoom' 


    def _watershed(self, viewer):
        """
        Execute watershed to split labels based on provided points. 
        Executes over one frame. 
        """
        # find the labels corresponding to the current points in the points layer
        labels = viewer.layers['Labels'].data
        image = viewer.layers['Image'].data
        points = viewer.layers['Points'].data
        points = np.round(points).astype(int)
        labels = watershed_split(image, labels, points)
        viewer.layers['Labels'].data = labels
        viewer.layers['Points'].data = np.empty((0, points.shape[1]), dtype=float)
        viewer.layers['Labels'].refresh()    
    

    def _select_colour(self, viewer):
        """
        Select colour for painting
        """
        if viewer.layers['Labels'].mode != 'pick': 
            viewer.layers['Labels'].mode = 'pick'  
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'  
    

    def _fill(self, viewer):
        """
        Switch napari labels layer to fill mode
        """
        if viewer.layers['Labels'].mode != 'fill':
            viewer.layers['Labels'].mode = 'fill' 
        else:
            viewer.layers['Labels'].mode = 'pan_zoom'


    def _paint(self, viewer):
        """
        Switch napari labels layer to paint mode
        """
        if viewer.layers['Labels'].mode != 'paint':  
            viewer.layers['Labels'].mode = 'paint'  
        else:  
            viewer.layers['Labels'].mode = 'pan_zoom'
    

    def _save(self, viewer): 
        # this damn thing that made me write this into a class :facepalm:
        # Perhaps there is a better way?
        """ 
        Save the annotated frame as a zarr file
        """
        output = viewer.layers['Labels'].data
        zarr.save_array(self.save_path, output)
        print("Labels saved at:")
        print(self.save_path)


# Split Objects
# -------------
def watershed_split(
                    image, 
                    labels, 
                    points, 
                    compactness=200, 
                    connectivity=octahedron(7)
                    ):
    """
    Split labels with using points as markers for watershed
    """
    points = np.round(points).astype(int)
    coords = tuple([points[:, i] for i in range(points.shape[1])])
    p_lab = labels[coords]
    p_lab = np.unique(p_lab)
    p_lab = p_lab[p_lab != 0]
    # generate a mask corresponding to the labels that need to be split
    mask = np.zeros(labels.shape, dtype=bool)
    for lab in p_lab:
        where = labels == lab
        mask = mask + where
    # split the labels using the points (in the masked image)
    markers = np.zeros(labels.shape, dtype=bool)
    markers[coords] = True
    markers = ndi.label(markers)
    markers = markers[0]
    new_labels = watershed(
                           image, 
                           markers=markers, 
                           mask=mask, 
                           compactness=compactness, 
                           connectivity=connectivity
                           )
    new_labels[new_labels != 0] += labels.max()
    # assign new values to the original labels
    labels = np.where(mask, new_labels, labels)
    return labels


# Execute
# -------
if __name__ == '__main__':
    path = '/Users/amcg0011/Data/pia-tracking/191113_IVMTR26_Inj3_cang_exp3.zarr'
    labs = '/Users/amcg0011/Data/pia-tracking/191113_IVMTR26_Inj3_cang_exp3_labels_t74_GT.zarr'
    correct_labels(path, labs, 74)
