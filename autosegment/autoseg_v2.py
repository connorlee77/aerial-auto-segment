import sys
sys.path.append('..')

import argparse

import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
from horizon.flir_boson_settings import I, D, P

import pickle
import tqdm
from scipy.spatial.transform import Rotation
from scipy.interpolate import NearestNDInterpolator
from scipy.signal import medfilt2d

import rasterio
import rasterio.plot
import rasterio.merge
import rasterio.mask
import pyproj
from PIL import ImageColor
import quaternion

from utils.utils import thermal2rgb
from utils.projections import (
    project_points,
    power_spacing,
    world2cam,
    create_world_grid,
    world_to_camera_coords,
    create_world_grid,
)

from utils.draw import (
    draw_overlay_and_labels,
    points_to_segmentation,
    generate_binary_mask,
    dynamic_world_color_map,
    chesapeake_cvpr_landcover_color_map,
    open_earth_map_landcover_color_map,
    project_elevation,
    project_prob
)

from gl_project import get_mask_mgl

##########################################################################
# Set path to flight sequence folder containing images and csv files.
##########################################################################
# DATA_PATH = '/data/onr-thermal/2022-12-20_Castaic_Lake/flight4'
# DATA_PATH = '/data/onr-thermal/2022-05-15_ColoradoRiver/flight3'
# DATA_PATH = '/data/onr-thermal/caltech_duck/ONR_2023-03-22-14-41-46'
# DATA_PATH = '/data/onr-thermal/big_bear/ONR_2022-05-08-11-23-59'
# DATA_PATH = '/data/onr-thermal/kentucky_river/flight3-1'


def colorize_dynamic_world_label(label):
    mapping = dynamic_world_color_map()

    color_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(0, 9):  # labels 0-8 inclusive
        color_label[label == i, :] = mapping[i]
    return color_label


def colorize_chesapeake_cvpr_landcover_label(label):
    mapping = chesapeake_cvpr_landcover_color_map()

    color_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(0, 7):  # labels 1-6 inclusive
        color_label[label == i, :] = mapping[i]
    return color_label


def colorize_open_earth_map_landcover_label(label):
    mapping = open_earth_map_landcover_color_map()

    color_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(0, 8):
        color_label[label == i, :] = mapping[i]
    return color_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/data/onr-thermal/2022-12-20_Castaic_Lake/flight4')
    parser.add_argument('--base_path', type=str, default='/data/microsoft_planetary_computer/outputs/preprocessed/')

    parser.add_argument('--epsg', type=str, default='epsg-32611', choices=['epsg-32618', 'epsg-32611', 'epsg-32616'])
    parser.add_argument('--place', type=str, default='castaic_lake',
                        choices=['colorado_river', 'castaic_lake', 'duck', 'big_bear_lake', 'kentucky_river'])
    parser.add_argument('--lulc_type', type=str, default='dynamicworld', choices=[
        'dynamicworld',
        'chesapeake_bay_swin_crossentropy_lc_naip_corrected',
        'chesapeake_bay_swin_crossentropy_lc_planet',
        'open_earth_map_unet_lc_naip_corrected',
        'open_earth_map_unet_lc_planet',
    ])

    parser.add_argument('--resolution', type=str, default='1.0', choices=['0.6', '1.0', '2.0', '3.0', '5.0', '10.0'])
    parser.add_argument('--refinement_type', type=str, default=None,
                        choices=['none', 'crf_naip_naip-nir', 'crf_naip_naip-nir_surface_height', 'crf_planet', 'crf_planet_surface_height'])

    parser.add_argument('--d3_type', type=str, default='dsm', choices=['dsm', 'dem', 'dem_1m'])

    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--image_list_to_run', type=str, default='labeled_cartd_autosegment.txt')

    args = parser.parse_args()
    args.refinement_type = None if args.refinement_type == 'none' else args.refinement_type
    print(args)

    images_to_run = []
    with open(args.image_list_to_run, 'r') as f:
        for line in f.readlines():
            images_to_run.append(line.strip())
    images_to_run = set(images_to_run)

    image_paths = sorted(glob.glob(os.path.join(args.data_path, 'images/thermal/*')))
    if len(images_to_run) > 0:
        print('Filtering {} images to run'.format(len(image_paths)))
        image_paths_to_run = []
        for img_path in tqdm.tqdm(image_paths):
            dataset, trajectory, _, _, name = img_path.split('/')[-5:]
            compare_name = os.path.join(dataset, trajectory, name.replace('tiff', 'png'))
            if compare_name in images_to_run:
                image_paths_to_run.append(img_path)
        image_paths = image_paths_to_run
        print('Running only images in image list: {} total images'.format(len(image_paths)))

    # Read the raster containing the class labels. This defaults to the dynamic world labels (10m).
    # with the option for other refinement types (e.g. CRF).
    if args.refinement_type is None:
        label_raster_path = os.path.join(args.base_path, args.epsg, args.place, args.lulc_type,
                                         args.resolution, 'mosaic.tiff')
    else:
        label_raster_path = os.path.join(args.base_path, args.epsg, args.place, args.lulc_type,
                                         args.resolution, args.refinement_type, 'mosaic.tiff')
    # Read the raster containing the digital surface map for 3D information
    dsm_path = os.path.join(args.base_path, args.epsg, args.place, args.d3_type, args.resolution, 'mosaic.tiff')

    assert os.path.exists(label_raster_path), 'Label raster does not exist for {}: {}'.format(
        args.place, label_raster_path)
    assert os.path.exists(dsm_path), '3d info {} does not exist for {}: {}'.format(args.d3_type, args.place, dsm_path)

    # Our data is in NED coordinates, so we need to subtract the baseline elevation to get the correct height.
    # This is in meters.
    dist_to_ground_plane = None
    if args.place == 'castaic_lake':
        baseline_elevation = 428.54
    elif args.place == 'colorado_river':
        baseline_elevation = 114  # 14m start above river level, account for this.
        dist_to_ground_plane = -14  # 14m start above river level, not included in csv file.
    elif args.place == 'duck':
        baseline_elevation = 0
    elif args.place == 'big_bear_lake':
        raise Exception('Big Bear Lake dataset does not have usable GPS data')
    elif args.place == 'kentucky_river':
        baseline_elevation = 160 # google earth says 178, but USGS dsm's say its 158-160m
        dist_to_ground_plane = -2  # 2m start above river level, not included in csv file.
    else:
        raise Exception('Unknown place: {}'.format(args.place))

    ##########################################################################
    # Create output folders
    ##########################################################################
    os.makedirs(args.output_dir, exist_ok=True)

    # Get colorize function
    if args.lulc_type == 'dynamicworld':
        colorize_func = colorize_dynamic_world_label
    elif 'chesapeake' in args.lulc_type:
        colorize_func = colorize_chesapeake_cvpr_landcover_label
    elif 'open_earth_map' in args.lulc_type:
        colorize_func = colorize_open_earth_map_landcover_label
    else:
        raise Exception('Unknown lulc type: {}'.format(args.lulc_type))

    ##########################################################################
    # Read csv of uav global/local pose
    ##########################################################################
    print('Reading data...')
    t0 = time.time()
    csv_path = os.path.join(args.data_path, "aligned.csv")
    assert os.path.exists(csv_path), 'Aligned CSV file (with imu/gps info) does not exist: {}'.format(csv_path)
    with open(csv_path, 'r') as f:
        header = 0
        while True:
            line = f.readline()
            if 'image' in line or (args.place == 'kentucky_river' and 'ir_file' in line):
                break
            header += 1
    alignment_data = pd.read_csv(csv_path, header=header)
    alignment_data = alignment_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    alignment_data.columns = alignment_data.columns.str.replace(' ', '')

    if args.place == 'kentucky_river':
        alignment_data.rename(columns={
            'ir_file': 'image',
            'uav_lat': 'camLLA_lat',
            'uav_lon': 'camLLA_lon',
            'ir_qw': 'camNED_qw',
            'ir_qx': 'camNED_qx',
            'ir_qy': 'camNED_qy',
            'ir_qz': 'camNED_qz',
            'ir_Z': 'camNED_D',
        }, inplace=True)

    print(alignment_data.head())
    t1 = time.time()
    print('{:3f} seconds to read csvs'.format(t1 - t0))

    ##########################################################################
    # Get rectified camera matrix
    ##########################################################################
    print('Creating new camera matrix...')
    H, W = (512, 640)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(I, D, (W, H), 0, (W, H))
    new_P = np.hstack([newcameramtx, np.zeros((3, 1))])

    ##########################################################################
    # Read raster data (dynamic world labels + dsm)
    ##########################################################################
    print('Reading rasters...')
    t0 = time.time()
    label_tiff_data = rasterio.open(label_raster_path)
    dsm = rasterio.open(dsm_path)

    # Create interpolators for the label and dsm rasters.
    if args.refinement_type is None:
        label_pkl_filepath = os.path.join('interpolators', args.place, args.lulc_type,
                                          args.resolution, 'standard', 'interp.pkl')
    else:
        label_pkl_filepath = os.path.join('interpolators', args.place, args.lulc_type,
                                          args.resolution, args.refinement_type, 'interp.pkl')
    dsm_pkl_filepath = os.path.join('interpolators', args.place, args.d3_type, args.resolution, 'interp.pkl')
    os.makedirs(os.path.dirname(label_pkl_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(dsm_pkl_filepath), exist_ok=True)
    if os.path.exists(label_pkl_filepath) and os.path.exists(dsm_pkl_filepath):
        with open(label_pkl_filepath, 'rb') as f:
            label_interp = pickle.load(f)

        with open(dsm_pkl_filepath, 'rb') as f:
            dsm_interp = pickle.load(f)
    else:
        label_array = label_tiff_data.read()
        dsm_array = dsm.read()
        dsm_array = medfilt2d(dsm_array.squeeze(), kernel_size=5)
        n_bands, height, width = label_array.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(label_tiff_data.transform, rows, cols)
        label_utm_grid = np.stack([xs, ys], axis=2).reshape(-1, 2)
        label_interp = NearestNDInterpolator(label_utm_grid, label_array.transpose(1,
                                                                                   2, 0).reshape(height * width, n_bands))

        with open(label_pkl_filepath, 'wb') as f:
            pickle.dump(label_interp, f, pickle.HIGHEST_PROTOCOL)

        height, width = dsm_array.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(dsm.transform, rows, cols)
        dsm_utm_grid = np.stack([xs, ys], axis=2).reshape(-1, 2)
        dsm_interp = NearestNDInterpolator(dsm_utm_grid, dsm_array.reshape(height * width, 1))

        with open(dsm_pkl_filepath, 'wb') as f:
            pickle.dump(dsm_interp, f, pickle.HIGHEST_PROTOCOL)

    t1 = time.time()
    print('{:3f} seconds to read rasters'.format(t1 - t0))

    # EPSG:4326 is lat/lng, which is what the drone GPS records in. Convert to UTM.
    crs = label_tiff_data.crs
    tform = pyproj.Transformer.from_crs("epsg:4326", "epsg:{}".format(crs.to_epsg()))

    # Setup camera matrices
    H, W = (512, 640)
    corrected_I, roi = cv2.getOptimalNewCameraMatrix(I, D, (W, H), 0, (W, H))

    ##########################################################################
    # Begin segmentation projection here
    ##########################################################################
    print('Starting segmentation estimation')
    for t, img_path in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):

        image_data = alignment_data[alignment_data['image'] == "images/thermal/{}".format(os.path.basename(img_path))]
        if len(image_data) == 0:
            print('Skipping {}, no pose info...'.format(img_path))
            continue

        coords = image_data[['camLLA_lat', 'camLLA_lon']].values.astype(float)[0]
        cam_xyzw = image_data[['camNED_qx', 'camNED_qy', 'camNED_qz', 'camNED_qw']].values.astype(float)[0]
        height = image_data[['camNED_D']].values.astype(float)[0, 0]

        if dist_to_ground_plane is None:
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

        img = cv2.imread(img_path, -1)
        img = thermal2rgb(img)  # Just stacks the thermal image into 3 channels.
        undistorted_image = cv2.undistort(img, I, D, None, newcameramtx)

        print("Distance to ground plane: {}".format(dist_to_ground_plane))
        z = height + dist_to_ground_plane
        r = Rotation.from_quat(cam_xyzw)
        yaw, pitch, roll = r.as_euler('ZYX', degrees=False)

        # Discretization of the world grid
        Nx = 250
        Ny = 100

        x_unit_vec, y_unit_vec, x_magnitudes, y_magnitudes, world_pts, xx, yy = create_world_grid(
            yaw,
            x_mag=10000,  # Forward
            y_mag=8000,  # side to side
            Nx=Nx,
            Ny=Ny,
            # Stuff compresses in far field, so create grid with less points in far field using exponential steps.
            exp_x=3,
            exp_y=3,
        )
        
        # Convert lat/lng of drone to UTM
        utm_e, utm_n = tform.transform(coords[0], coords[1])
        rows, cols = rasterio.transform.rowcol(label_tiff_data.transform, xs=utm_e, ys=utm_n)

        ptA_utm = np.array([utm_e, utm_n])
        ptA_rc = np.array([cols, rows])

        y_grid = y_unit_vec.reshape(2, 1) * yy.reshape(1, Nx * Ny)
        x_grid = ptA_utm.reshape(2, 1) + x_unit_vec.reshape(2, 1) * xx.reshape(1, Nx * Ny)
        utm_grid = x_grid + y_grid
        utm_grid = utm_grid.reshape(2, Ny, Nx).transpose(2, 1, 0)

        t1 = time.time()
        sampled_labels = label_interp(utm_grid.reshape(-1, 2))
        sampled_z = dsm_interp(utm_grid.reshape(-1, 2))
        t2 = time.time()
        print('{:3f} seconds to interpolate'.format(t2 - t1))

        t1 = time.time()
        # Offset elevation map for the correct elevation
        world_coord_z = np.clip(sampled_z.reshape(Nx * Ny, 1) - baseline_elevation, 0, None)
        word_coord_pts = np.concatenate([world_pts.reshape(Nx * Ny, 2), world_coord_z], axis=1)
        n_lulc_classes = sampled_labels.shape[1]

        # Account for NED coordinate frame (x: forward, y: right, z: down)
        world_coord_labels = sampled_labels.reshape(Nx * Ny, n_lulc_classes)
        # if args.place in ('kentucky_river'):
        #     # z is positive here
        #     word_coord_pts[:, 2] = word_coord_pts[:, 2] - z
        # else:
        #     # z is negative here
        #     word_coord_pts[:, 2] = -word_coord_pts[:, 2] - z
        word_coord_pts[:, 2] = -word_coord_pts[:, 2] - z

        # word_coord_pts are the 3D labels (x: forward, y: right, z: down)
        camera_pts = world_to_camera_coords(cam_xyzw, word_coord_pts).T
        labeled_camera_pts = np.concatenate([camera_pts, world_coord_labels[:, -1].reshape(-1, 1)], axis=1)
        name = os.path.basename(img_path).split('.')[0]
        # Transform to camera coordinates. Only used to directly project 3d points to image frame.
        # Not needed for OpenGL/ModernGL projection
        # Xn = world2cam(cam_xyzw, new_P, word_coord_pts)
        # # Project points onto image. Not needed for OpenGL/ModernGL projection
        # original_img, masked_img, pts_img = draw_overlay_and_labels(
        #     undistorted_image,
        #     points=Xn,
        #     labels=world_coord_labels[:, -1],
        #     color_map=dynamic_world_color_map(),
        # )

        # cv2.imwrite('temp/{}.png'.format(name), original_img)
        # # cv2.imwrite('outputs/{}_autoseg.png'.format(name), masked_img)
        # cv2.imwrite('temp/{}_pts.png'.format(name), pts_img)

        if args.place == 'kentucky_river':
            original_img = undistorted_image
        else:
            original_img = cv2.rotate(undistorted_image, cv2.ROTATE_180)

        t1 = time.time()

        # NOTE: This is the OpenGL/ModernGL projection. This is the one we use for the paper.
        mgl_mask = get_mask_mgl(labeled_camera_pts.T, corrected_I, colorize_func=colorize_func)

        # check that the channels are all the same
        assert np.all(mgl_mask[:, :, 0] == mgl_mask[:, :, 1])
        # The mask has 3 channels only for convenience as code was written to render RGB images. The channels are the same.
        # 0 is reserved for "unrendered" and mask is 1-indexed. Subtract 1 to get 0-indexed mask and -1 is not considered in mask.
        mgl_mask = mgl_mask[:, :, 0]
        mgl_mask = mgl_mask - 1
        mgl_mask[mgl_mask < 0] = 255  # this is probably sky

        mgl_colorized_mask = colorize_func(mgl_mask)
        t2 = time.time()
        print('{:3f} seconds to render mask'.format(t2 - t1))

        name = os.path.basename(img_path).split('.')[0]
        # cv2.imwrite('original_output/{}_autoseg_refined.png'.format(name), cv2.cvtColor(refined_mask, cv2.COLOR_RGB2BGR))
        overlay = cv2.addWeighted(original_img, 0.7, mgl_colorized_mask, 0.3, 0)
        cv2.imwrite('{}/{}_overlay.png'.format(args.output_dir, name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite('{}/{}_color_mask.png'.format(args.output_dir, name),
                    cv2.cvtColor(mgl_colorized_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite('{}/{}_mask.png'.format(args.output_dir, name), mgl_mask)
