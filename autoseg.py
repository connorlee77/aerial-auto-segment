import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
import shutil
from horizon.flir_boson_settings import I, D, P

import pickle
import tqdm
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

import rasterio
import rasterio.plot
import rasterio.merge
import rasterio.mask
import pyproj
from PIL import ImageColor
import shapely
from shapely import Polygon
import skimage 

from utils.utils import thermal2rgb
from utils.projections import (
    project_points, 
    power_spacing, 
    world2cam, 
    create_world_grid,
)

from utils.draw import (
    draw_overlay_and_labels, 
    points_to_segmentation, 
    generate_binary_mask,
    colorize_dynamic_world_label,
    HEX_COLORS,
    dynamic_world_color_map,
)



##########################################################################
### Set path to flight sequence folder containing images and csv files.
##########################################################################
# DATA_PATH = '/media/hdd2/data/caltech_duck/ONR_2023-03-22-14-41-46'
# DATA_PATH = '/home/carson/data/thermal/2022-12-20_Castaic_Lake/flight4'
DATA_PATH = '/data/onr-thermal/2022-12-20_Castaic_Lake/flight4'

BASELINE_ELEVATION = 428.54 # Water elevation of Castaic Lake, Dec. 22, 2022
LABEL_RASTER_PATH = 'label_mosaic_v2.tiff'
DSM_PATH = '/data/microsoft_planetary_computer/dsm/castaiclake/mosaic/mosaic.tiff'


##########################################################################
### Create output folders
##########################################################################
if os.path.exists('outputs') and os.path.isdir('outputs'):
    shutil.rmtree('outputs')
os.makedirs('outputs')

color_map = dynamic_world_color_map()


##########################################################################
### Read csv of uav global/local pose
##########################################################################
print('Reading data...')
t0 = time.time()
alignment_data = pd.read_csv(os.path.join(DATA_PATH, "aligned.csv"), header=13)
alignment_data = alignment_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
alignment_data.columns = alignment_data.columns.str.replace(' ', '')
t1 = time.time()
print('{:3f} seconds to read csvs'.format(t1 - t0))


##########################################################################
### Get rectified camera matrix
##########################################################################
print('Creating new camera matrix...')
H, W = (512, 640)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(I, D, (W, H), 0, (W, H))
new_P = np.hstack([newcameramtx, np.zeros((3,1))])


##########################################################################
### Read raster data (dynamic world labels + dsm)
##########################################################################
t0 = time.time()
label_tiff_data = rasterio.open(LABEL_RASTER_PATH)
dsm = rasterio.open(DSM_PATH)

if os.path.exists('dw_interp.pkl') and os.path.exists('dsm_interp.pkl'):
    with open('dw_interp.pkl', 'rb') as f:
        label_interp = pickle.load(f)
    
    with open('dsm_interp.pkl', 'rb') as f:
        dsm_interp = pickle.load(f)

else:
    label_array = label_tiff_data.read()
    dsm_array = dsm.read()

    print(label_array.shape)
    n_bands, height, width = label_array.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(label_tiff_data.transform, rows, cols)
    label_utm_grid = np.stack([xs, ys], axis=2).reshape(-1, 2)
    label_interp = NearestNDInterpolator(label_utm_grid, label_array.transpose(1, 2, 0).reshape(height*width, n_bands))

    with open('dw_interp.pkl', 'wb') as f:
        pickle.dump(label_interp, f, pickle.HIGHEST_PROTOCOL)

    print(dsm_array.shape)
    n_bands, height, width = dsm_array.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(dsm.transform, rows, cols)
    dsm_utm_grid = np.stack([xs, ys], axis=2).reshape(-1, 2)
    dsm_interp = NearestNDInterpolator(dsm_utm_grid, dsm_array.transpose(1, 2, 0).reshape(height*width, 1))
    with open('dsm_interp.pkl', 'wb') as f:
        pickle.dump(dsm_interp, f, pickle.HIGHEST_PROTOCOL)
t1 = time.time()
print('{:3f} seconds to read rasters'.format(t1 - t0))

crs = label_tiff_data.crs
tform = pyproj.Transformer.from_crs("epsg:4326", "epsg:{}".format(crs.to_epsg()))


##########################################################################
### Begin segmentation projection here
##########################################################################
print('Starting segmentation estimation')
image_paths = sorted(glob.glob(os.path.join(DATA_PATH, 'images/thermal/*')))
for t, img_path in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):
    if t < 2000:
        continue
    # img_path = os.path.join(DATA_PATH, 'images/thermal/thermal-50000.tiff')
    # img_path = os.path.join(DATA_PATH, 'images/thermal/thermal-20000.tiff')
    # img_path = '/home/carson/data/thermal/2022-05-15_ColoradoRiver/flight3/images/thermal/thermal-02500.tiff'
    # output_path = 'outputs/{}'.format(os.path.basename(img_path).replace('tiff', 'png'))

    image_data = alignment_data[alignment_data['image'] == "images/thermal/{}".format(os.path.basename(img_path))]
    if len(image_data) == 0:
        print('Skipping {}, no pose info...'.format(img_path))
        continue
    
    coords = image_data[['camLLA_lat', 'camLLA_lon']].values.astype(float)[0]
    cam_xyzw = image_data[['camNED_qx', 'camNED_qy', 'camNED_qz', 'camNED_qw']].values.astype(float)[0]
    height = image_data[['camNED_D']].values.astype(float)[0, 0]
    dist_to_ground_plane = image_data[['riverNED_Z']].values.astype(float)[0, 0]
    if np.isnan(coords).any():
        print('Skipping {}, lat/lng (camLLA_lat/lng) has NaNs...'.format(img_path))
        continue
    if np.isnan(cam_xyzw).any():
        print('Skipping {}, quaternion has NaNs...'.format(img_path))
        continue
    if np.isnan(height).any():
        print('Skipping {}, uav altitude (camNED_D) has NaNs...'.format(img_path))
        continue
    if np.isnan(dist_to_ground_plane).any():
        print('Skipping {}, uav to ground distance (riverNED_Z) has NaNs...'.format(img_path))
        continue

    print('Processing image: {}'.format(img_path))
    img = cv2.imread(img_path, -1)
    img = thermal2rgb(img)
    undistorted_image = cv2.undistort(img, I, D, None, newcameramtx)

    z = height + dist_to_ground_plane
    r = Rotation.from_quat(cam_xyzw)
    yaw, pitch, roll =  r.as_euler('ZYX', degrees=False)
    
    N = 200

    t0 = time.time()
    x_unit_vec, y_unit_vec, x_magnitudes, y_magnitudes, world_pts = create_world_grid(
        yaw, 
        x_mag=8000,
        y_mag=3000,
        N=N,
        exp_x=5,
        exp_y=5,
    )
    t1 = time.time()
    print('{:3f} seconds to create world grid'.format(t1 - t0))

    utm_e, utm_n = tform.transform(coords[0], coords[1])
    rows, cols = rasterio.transform.rowcol(label_tiff_data.transform, xs=utm_e, ys=utm_n)

    ptA_utm = np.array([utm_e, utm_n])
    ptA_rc = np.array([cols, rows])

    y_grid = y_unit_vec.reshape(2, 1) * y_magnitudes.reshape(1, N)
    x_grid = ptA_utm.reshape(2, 1) + x_unit_vec.reshape(2, 1) * x_magnitudes.reshape(1, N)
    utm_grid = x_grid.T.reshape(N, 1, 2) + y_grid.T.reshape(1, N, 2)

    # t0 = time.time()
    # sampled_labels = rasterio.sample.sample_gen(label_tiff_data, xy=utm_grid.reshape(-1, 2))
    # sampled_z = rasterio.sample.sample_gen(dsm, xy=utm_grid.reshape(-1, 2))
    # t1 = time.time()
    # print('{:3f} seconds to sample from rasters'.format(t1 - t0))

    t0 = time.time()
    print(utm_grid.shape)
    sampled_labels = label_interp(utm_grid.reshape(-1, 2))
    sampled_z = dsm_interp(utm_grid.reshape(-1, 2))
    t1 = time.time()
    print('{:3f} seconds to sample from array'.format(t1 - t0))

    t0 = time.time()
    # world_coord_label_map = np.zeros((N*N, 4)) # x, y, z, label
    # for i, (val, h) in enumerate(zip(sampled_labels, sampled_z)):
    #     print(type(val), h)
    #     world_coord_label_map[i, 0:2] = world_pts[i]
    #     world_coord_label_map[i, 2] = np.clip(h - BASELINE_ELEVATION, 0, None)
    #     world_coord_label_map[i, 3] = val[-1] 
    #     exit(0)

    world_coord_z = np.clip(sampled_z.reshape(N*N, 1) - BASELINE_ELEVATION, 0, None)
    world_coord_labels = sampled_labels[:,-1].reshape(N*N, 1)
    world_coord_label_map = np.concatenate([world_pts.reshape(N*N, 2), world_coord_z, world_coord_labels], axis=1)
    t1 = time.time()
    print('{:3f} seconds to label world coordinates'.format(t1 - t0))

    # Label points in back first
    ind = np.argsort(world_coord_label_map[:,0])[::-1]
    world_coord_label_map = world_coord_label_map[ind]
    surface_elevation = np.copy(world_coord_label_map[:, 2])
    world_coord_label_map[:, 2] = -world_coord_label_map[:, 2] - z

    ### SARASWATI: world_coord_label_map are the 3D labels (x: forward, y: right, z: down)

    Xn = world2cam(cam_xyzw, new_P, world_coord_label_map)

    original_img, masked_img, pts_img = draw_overlay_and_labels(
        undistorted_image, 
        points=Xn, 
        labels=world_coord_label_map[:,3], 
        color_map=color_map
    )

    name = os.path.basename(img_path).split('.')[0]
    cv2.imwrite('outputs/{}.png'.format(name), original_img)
    cv2.imwrite('outputs/{}_autoseg.png'.format(name), masked_img)
    cv2.imwrite('outputs/{}_pts.png'.format(name), pts_img)

##########################################################################
##########################################################################
##########################################################################