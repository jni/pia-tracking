import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import time
import yaml
from pathlib import Path
from nd2reader import ND2Reader
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import math
from scipy.spatial.transform import Rotation as Rot


def get_datetime():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_nd2_files(data_path):
    return list(Path(data_path).rglob('*.nd2'))

def get_objectDF_files(data_path):
    return list(Path(data_path).rglob('*.df.pkl'))

def get_nd2_info(nd2_data):
    metadata_dict = dict(file=str(nd2_data.filename), px_microns=nd2_data.metadata['pixel_microns'])
    metadata_to_save = ['x', 'y', 'c', 't', 'z']
    metadata_dict.update(**{ m:nd2_data._sizes[m] for m in metadata_to_save})
    metadata_dict.update(**{ 'channel_'+ str(i):c for i, c in enumerate(nd2_data.metadata['channels'])})
    metadata_dict.update(frame_rate = float(nd2_data.frame_rate))
    metadata_dict.update(roi_t = float(np.mean(nd2_data.metadata['rois'][0]['timepoints'])))
    metadata_dict.update(roi_x = int(nd2_data.metadata['rois'][0]['positions'][0][1]))
    metadata_dict.update(roi_y = int(nd2_data.metadata['rois'][0]['positions'][0][0]))
    metadata_dict.update(roi_size = float(nd2_data.metadata['rois'][0]['sizes'][0][0]))
    return metadata_dict


def nd2_info_to_df(data_path):
    nd2_infos=[]
    nd2_files = get_nd2_files(data_path)
    for file in nd2_files:
        with ND2Reader(file) as nd2_data:
            nd2_infos.append(get_nd2_info(nd2_data))
    return pd.DataFrame(nd2_infos)

def get_nd2_vol(nd2_data, c, frame):
    nd2_data.default_coords['c']=c
    nd2_data.bundle_axes = ('y', 'x', 'z')
    v = nd2_data.get_frame(frame)
    v = np.array(v)
    return v

def dict_fix_numpy(dict_in):
    dict_out = dict()
    for di in dict_in.keys():
        if type(dict_in[di]) == np.float64:
            dict_out[di] = float(dict_in[di]) 
        elif type(dict_in[di]) == np.int64:
            dict_out[di] = int(dict_in[di])
        else:
            dict_out[di] = dict_in[di]
    return dict_out

def create_dog_func(s1, s2):
    def dog_func(image):
        image_dog = cv2.GaussianBlur(image.astype('float'),(0,0), s1) - cv2.GaussianBlur(image.astype('float'), (0,0), s2)
        return image_dog
    return dog_func

def denoise(image):
    res_im = cv2.fastNlMeansDenoising(image, None, 6, 7, 20)
    return res_im

def array_to_int8(arr):
    arr8 = arr-arr.min()
    arr8 = ((arr8/arr8.max())*255).astype('uint8')
    return arr8

def vol_to_images(vol):
    im_list= [vol[...,i] for i in range(vol.shape[-1])]
    return im_list

def images_to_vol(images):
    vol = np.stack(images, axis=2)
    return vol

def image_func_to_vol(func):
    def vol_func(vol):
        images = vol_to_images(vol)
        res_images = list(map(func, images))
        res_vol = images_to_vol(res_images)
        return res_vol
    return vol_func

def imshow(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray')

def imshow_spectral(image, **kwargs):
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='nipy_spectral', **kwargs)

def imshow_prism(image, **kwargs):
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='prism', **kwargs)

def median_threshold(image, adjust=10):
    image_median = np.median(image)
    threshold = image_median + adjust
    return threshold, image>threshold

def voting_threshold(image, adjust=10):
    voting = np.bincount(image.ravel()).argmax()
    threshold = voting + adjust
    return threshold, image>threshold

def interpolate_volume(vol, zfactor=4):
    int_vol = ndi.zoom(thv, (1, 1, zfactor), order=0)
    return int_vol

def get_object_labels(vol):
    labels, num_objects = ndi.label(vol, structure=np.ones((3,3,3)))
    return num_objects, labels

def get_object_labels_watershed(vol):
    distance = ndi.distance_transform_edt(vol)
    local_maxi = peak_local_max(distance, indices=False, min_distance=3,labels=vol)
    markers, num_objects = ndi.label(local_maxi, structure=np.ones((3,3,3)))
    labels = watershed(-distance, markers, mask=vol)
    return num_objects, labels

def get_object_labels_watershed2(vol, th_vol):
    distance = ndi.distance_transform_edt(th_vol)
    local_maxi = peak_local_max(vol, indices=False, min_distance=3,labels=th_vol)
    markers, num_objects = ndi.label(local_maxi, structure=np.ones((3,3,3)))
    labels = watershed(-vol, markers, mask=th_vol)
    return num_objects, labels

def get_object_labels_watershed3(vol, th_vol):
    distance = distance_transform_xy(th_vol)
    local_maxi = peak_local_max(vol, indices=False, min_distance=3,labels=th_vol)
    markers, num_objects = ndi.label(local_maxi, structure=np.ones((3,3,3)))
    labels = watershed(-vol, markers, mask=th_vol)
    return num_objects, labels

def get_object_labels_watershed4(vol, vol_th):
    image_list = vol_to_images(vol)
    vol_dg = images_to_vol(list(map(dog, image_list)))
    vol_dg_th = vol_dg >0.1
    vol_dg = array_to_int8(vol_dg)
    local_maxi = peak_local_max(vol_dg, indices=False, min_distance=3,labels=vol_dg_th)
    markers, num_objects = ndi.label(local_maxi, structure=np.ones((3,3,3)))
    labels = watershed(-vol_dg, markers, mask=vol_th)
    return num_objects, labels

def save_yaml(dict_to_yaml, filename):
    with open(filename, 'w') as file:
        _ = yaml.dump(dict_to_yaml, file)


class IvmObjects:
    def __init__(self, conf):
        self.conf = conf
        self.inspect_steps = {}
    
    def add_nd2info(self, nd2info):
        self.conf['nd2info'] = dict_fix_numpy(nd2info.to_dict())
    
    def _process_frame(self, frame):
        # load volume
        #d2_data = ND2Reader(self.nd2_file)
        with ND2Reader(self.nd2_file) as nd2_data:
            v = get_nd2_vol(nd2_data, self.conf['object_channel'], frame)
            self.inspect_steps[0]=dict(name = 'original_volume', data = v) 
            v = array_to_int8(v)

            # denoise images
            vi = vol_to_images(v)
            vi_dn = list(map(denoise, vi))
            v_dn = images_to_vol(vi_dn)
            v_dn = array_to_int8(v_dn)
            self.inspect_steps[1]=dict(name = 'denoised_volume', data = v_dn) 
            #th, v_th = voting_threshold(v, adjust=8)

            # difference of gaussian
            v_dni= vol_to_images(v_dn)
            dog = create_dog_func(self.conf['dog_sigma1'], self.conf['dog_sigma2'])
            v_dg = images_to_vol(list(map(dog, v_dni)))
            self.inspect_steps[2]=dict(name = 'dog_volume', data = v_dg) 

            # threshold
            v_dg_th = v_dg > self.conf['threshold']
            v_dg = array_to_int8(v_dg)
            self.inspect_steps[3] = dict(name = 'threshold_volume', data = v_dg_th) 

            # watershed and create labels
            local_maxi = peak_local_max(v_dg, indices=False, min_distance=self.conf['peak_min_dist'],labels=v_dg_th)
            markers, num_objects = ndi.label(local_maxi, structure=np.ones((3,3,3)))
            #v_labels = watershed(-v_dg, markers, mask=v_dg_th)
            v_labels = watershed(-v_dg, markers, mask=v_dg_th,compactness=1)
            self.inspect_steps[4] = dict(name = 'labels_volume', data = v_labels) 

            # extract info from labels
            labels_idx = np.arange(1, v_labels.max())
            label_pos=ndi.measurements.center_of_mass(v_dg_th, v_labels, labels_idx)
            df=pd.DataFrame(label_pos)

            #collect data for inspection
            if self.conf['process_type'] == 'single_thread':
                self.inspect_steps[0] = dict(name = 'original_volume', data = v) 
                self.inspect_steps[1] = dict(name = 'denoised_volume', data = v_dn) 
                self.inspect_steps[2] = dict(name = 'dog_volume', data = v_dg) 
                self.inspect_steps[3] = dict(name = 'threshold_volume', data = v_dg_th) 
                self.inspect_steps[4] = dict(name = 'labels_volume', data = v_labels) 



            #makes a dataframe with all coordinates
            df.columns=['x', 'y', 'z']

            # adjust xs, ys to centrer roi
            if self.conf['center_roi']:
                adjust_x = self.conf['nd2info']['roi_x'] 
                adjust_y = self.conf['nd2info']['roi_y'] 
            else: 
                adjust_x = 0
                adjust_y = 0
           
                
            df['xs'] = df['x'] * self.conf['nd2info']['px_microns'] - adjust_x 
            df['ys'] = df['y'] * self.conf['nd2info']['px_microns'] - adjust_y 
            df['zs'] = df['z'] * self.conf['z_dist']
            
            if self.conf['rotate']:
                #theta = np.radians(self.conf['rotate_angle'])
                #df['xxs'] = df['xs']*np.cos(theta) + df['ys']*np.sin(theta)
                #df['yys'] = df['ys']*np.cos(theta) - df['xs']*np.sin(theta)
                
                rot = Rot.from_euler('z', -self.conf['rotate_angle'], degrees=True)
                xyz = df[['xs', 'ys', 'zs']].to_numpy()
                xyz_rot = rot.apply(xyz)
                df['xs'], df['ys'] =  xyz_rot[:,0], xyz_rot[:,1]
                
            
            df.insert(0, 'frame',frame)
            df.insert(1, 'time', frame/self.conf['nd2info']['frame_rate'])
            df.insert(0, 'path', self.nd2_file)

            df['size']=ndi.measurements.sum(v_dg_th, v_labels, labels_idx)
            df['int_mean']=ndi.measurements.mean(v, v_labels, labels_idx)
            df['int_max']=ndi.measurements.maximum(v, v_labels, labels_idx)

            #df['c']=c

            intensity_channels = self.conf['intensity_channels']

            for c in intensity_channels:
                v_int = get_nd2_vol(nd2_data, c, frame)
                #v_int=ndimage.zoom(imf.get_vol(t=t, c=c2), (1,1,4),order=0)

                df['c' + str(c) + '_mean']=ndi.measurements.mean(v_int, v_labels, labels_idx)
                df['c' + str(c) + '_max']=ndi.measurements.maximum(v_int, v_labels, labels_idx)
                
        
        return df


    def process_file(self, nd2_file, frames):
        starttime = time.time()
        print('Starting :', nd2_file, '...',end='')

        self.nd2_file = nd2_file


        if self.conf['process_type'] == 'multi_thread':
            with ThreadPoolExecutor(max_workers=self.conf['multi_workers'] ) as executor:
                futures = executor.map(self._process_frame, frames)
            df_obj_frames = list(futures)

        elif self.conf['process_type'] == 'multi_process':
            with ProcessPoolExecutor(max_workers=self.conf['multi_workers']) as executor:
                futures = executor.map(self._process_frame, frames)
            df_obj_frames = list(futures)

        else:
            df_obj_frames = list(map(self._process_frame, frames))

        df_obj = pd.concat(df_obj_frames, ignore_index=True, axis=0)
        
        #post process dataframe
        df_obj['zf'] = df_obj['zs'] - np.percentile(df_obj['zs'], 2)
        df_obj.insert(0, 'pid',  df_obj.reset_index()['index'])

        print('OK')
        print('Processed in {0:.2f} seconds. Found {1:} platelets.'.format((time.time()-starttime), len(df_obj.index)))
        return df_obj

