# IPython log file
# View the nd2 platelet data in napari using dask

import napari
from nd2reader import ND2Reader

# viewing a single timepoint

# get_ipython().run_line_magic('cd', '~/Dropbox/share-files/')
nd2_data = ND2Reader('200519_IVMTR69_Inj4_dmso_exp3.nd2')
object_channel = 2
def get_nd2_vol(nd2_data, c, frame):
    nd2_data.default_coords['c']=c
    nd2_data.bundle_axes = ('y', 'x', 'z')
    v = nd2_data.get_frame(frame)
    v = np.array(v)
    return v

fram = get_nd2_vol(nd2_data, object_channel, 70)
napari.view_image(fram, scale=[1, 1, 4], ndisplay=3)


# adding all timepoints using dask and viewing the whole volume
from dask import delayed
import toolz as tz
nd2vol = tz.curry(get_nd2_vol)
arr = da.stack(
    [da.from_delayed(delayed(nd2vol(nd2_data, 2))(i),
     shape=fram.shape,
     dtype=fram.dtype)
     for  i in range(193)]  # note hardcoded n-timepoints
     )
napari.view_image(arr, scale=[1, 1, 1, 4])
