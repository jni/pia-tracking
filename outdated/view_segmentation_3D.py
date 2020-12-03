import fl
import importlib
import napari
from parser import custom_parser, hardcoded_paths


# Parse args
# ---------
parser = custom_parser()
args = parser.parse_args()
if args.name:
    data_path = hardcoded_paths(args.name, __file__)['data_path']
else:
    data_path = args.image


# Segmentation workflow
# ---------------------
# single frame segmentation from 1(4)_Process_images.ipynb 
importlib.reload(fl)
df_files= fl.nd2_info_to_df(data_path)
conf = dict(
            process_type = 'single_thread', # USE ONLY 'single_thread' for inspection
            multi_workers = 7, # dont use too many
            object_channel = 2,
            intensity_channels = [0, 1],
            dog_sigma1 = 1.7,
            dog_sigma2 = 2.0,
            threshold = 0.15,
            peak_min_dist = 3,
            z_dist = 2,
            center_roi = True,
            rotate = True,
            rotate_angle = 45,
            )
fileId = 0
frame = 70
ivmObjects = fl.IvmObjects(conf)
file, frames = df_files.iloc[fileId][['file', 't']]
ivmObjects.add_nd2info(df_files.iloc[fileId]) #add nd2-file info to conf
df_obj_insp = ivmObjects.process_file(file, [frame])
insp_steps = ivmObjects.inspect_steps


# View in 3D
# ----------
# View the saved steps 
with napari.gui_qt():
    viewer = napari.view_image(insp_steps[0]['data'], name=insp_steps[0]['name'], scale=[1, 1, 4]) # original
    viewer.add_image(insp_steps[1]['data'], name=insp_steps[1]['name'], scale=[1, 1, 4])
    viewer.add_image(insp_steps[2]['data'], name=insp_steps[2]['name'], scale=[1, 1, 4])
    viewer.add_labels(insp_steps[3]['data'], name=insp_steps[3]['name'], scale=[1, 1, 4])
    viewer.add_labels(insp_steps[4]['data'], name=insp_steps[4]['name'], scale=[1, 1, 4])