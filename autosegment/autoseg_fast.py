import sys

sys.path.append('..')

import argparse
import glob
import os
import pickle
import shutil
import time

import cv2
import moderngl
import numpy as np
import pandas as pd
import pyproj
import pyvista as pv
import rasterio
import rasterio.mask
import rasterio.merge
import rasterio.plot
import shapely
import skimage
import tqdm
from PIL import Image, ImageColor
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from scipy.signal import medfilt2d
from scipy.spatial.transform import Rotation
from shapely import Polygon

from horizon.flir_boson_settings import D, I, P
from utils.draw import (chesapeake_cvpr_landcover_color_map,
                        draw_overlay_and_labels, dynamic_world_color_map,
                        generate_binary_mask,
                        open_earth_map_landcover_color_map,
                        points_to_segmentation, project_elevation,
                        project_prob)
from utils.projections import (create_world_grid, power_spacing,
                               project_points, world2cam,
                               world_to_camera_coords)
from utils.utils import thermal2rgb
import startinpy

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

    h = len(label)
    color_label = np.zeros((h, 3), dtype=np.uint8)
    for i in range(0, 9):  # labels 0-8 inclusive
        color_label[label == i, :] = mapping[i]

    return color_label / 255


def colorize_chesapeake_cvpr_landcover_label(label):
    mapping = chesapeake_cvpr_landcover_color_map()

    h = len(label)
    color_label = np.zeros((h, 3), dtype=np.uint8)
    for i in range(0, 7):  # labels 1-6 inclusive
        color_label[label == i, :] = mapping[i]

    return color_label / 255


def colorize_open_earth_map_landcover_label(label):
    mapping = open_earth_map_landcover_color_map()

    h = len(label)
    color_label = np.zeros((h, 3), dtype=np.uint8)
    for i in range(0, 8):
        color_label[label == i, :] = mapping[i]

    return color_label / 255


def glOrtho(left, right, bottom, top, near, far):
    tx = -(right + left) / (right - left)
    ty = -(top + bottom) / (top - bottom)
    tz = -(far + near) / (far - near)
    x = np.array([
        [2 / (right - left), 0, 0, tx],
        [0, 2 / (top - bottom), 0, ty],
        [0, 0, -2 / (far - near), tz],
        [0, 0, 0, 1],
    ], dtype='f4')
    return x


def get_mask_mgl(pts, I, colorize_func=None):
    '''
        Renders 3d points and labels in a forward facing image. 
        Assumes that pts are already in the camera coordinate frame.

        Args:
            pts: 4xN array of (x, y, z, label)
            I: 3x3 intrinsics matrix (corrected for distortion)
            color_map: mapping from label id to rgb color array. 
        Returns:
            image: np array image of the rendered points
     '''

    near = 0.1
    far = 10000
    rows = 512
    cols = 640
    A = -(near + far)
    B = near * far

    K_gl = np.array([
        [I[0, 0], 0, -I[0, 2], 0],
        [0, I[1, 1], -I[1, 2], 0],
        [0, 0, near + far, near * far],
        [0, 0, -1, 0],
    ], dtype='f4')
    NDC = glOrtho(0, cols, rows, 0, near, far)

    P_gl = NDC @ K_gl

    xyz = pts[[1, 2, 0], :].T
    xyz[:, 2] *= -1
    xyz[:, 0] *= -1

    label = pts[3, :].T
    color = colorize_func(label)

    cloud = pv.PolyData(xyz)
    surf = cloud.delaunay_2d()
    vertex_normals = surf.point_normals
    vertices = surf.points
    vertex_colors = color

    faces = surf.faces.reshape(-1, 4)
    triangles = faces[:, 1:]
    t1 = time.time()
    with moderngl.create_context(standalone=True, backend='egl') as ctx:

        prog = ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 proj;

                in vec3 in_vert;
                in vec3 in_color;

                flat out vec3 v_color;
                out vec4 pose;

                void main() {
                    v_color = in_color;
                    pose = proj*vec4(in_vert, 1.0);
                    gl_Position = proj*vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                flat in vec3 v_color;
                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            ''',
            varyings=['pose']
        )

        ctx.enable(moderngl.DEPTH_TEST)
        # ctx.provoking_vertex = moderngl.LAST_VERTEX_CONVENTION

        P_gl = np.ascontiguousarray(P_gl.T)
        prog['proj'].write(P_gl)

        vertices_info = np.hstack([vertices, vertex_colors]).astype('f4')

        fbo = ctx.framebuffer(
            color_attachments=ctx.texture((640, 512), 3, samples=0),
            depth_attachment=ctx.depth_texture((640, 512), samples=0)
        )
        fbo.use()

        # fbo = ctx.simple_framebuffer((640, 512))
        # fbo.use()

        ibo = ctx.buffer(triangles.astype('i4'))
        vbo = ctx.buffer(vertices_info.astype('f4'))
        vao = ctx.vertex_array(
            prog,
            [(vbo, '3f 3f', 'in_vert', 'in_color')],
            ibo,
        )

        vao.render(moderngl.TRIANGLES)
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, 1)
    t2 = time.time()
    print('rendering time: ', t2 - t1)
    return np.asarray(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/data/onr-thermal/2022-12-20_Castaic_Lake/flight4')
    parser.add_argument('--base_path', type=str, default='/data/microsoft_planetary_computer/outputs/preprocessed/')

    parser.add_argument('--epsg', type=str, default='epsg-32611', choices=['epsg-32618', 'epsg-32611', 'epsg-32616'])
    parser.add_argument('--place', type=str, default='castaic_lake',
                        choices=['colorado_river', 'castaic_lake', 'duck', 'big_bear', 'kentucky_river'])
    parser.add_argument('--lulc_type', type=str, default='dynamicworld')
    parser.add_argument('--resolution', type=str, default='1.0', choices=['0.6', '1.0', '2.0', '3.0', '5.0', '10.0'])
    parser.add_argument('--refinement_type', type=str, default=None, choices=[None, 'crf_naip_naip-nir'])

    parser.add_argument('--d3_type', type=str, default='dsm')

    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()
    print(args)

    if args.refinement_type is None:
        label_raster_path = os.path.join(args.base_path, args.epsg, args.place, args.lulc_type,
                                         args.resolution, 'mosaic.tiff')
    else:
        label_raster_path = os.path.join(args.base_path, args.epsg, args.place, args.lulc_type,
                                         args.resolution, args.refinement_type, 'mosaic.tiff')
    dsm_path = os.path.join(args.base_path, args.epsg, args.place, args.d3_type, args.resolution, 'mosaic.tiff')

    if args.place == 'castaic_lake':
        baseline_elevation = 428.54
    elif args.place == 'colorado_river':
        baseline_elevation = 114
    elif args.place == 'duck':
        baseline_elevation = 0
    elif args.place == 'big_bear':
        baseline_elevation = 2058
    elif args.place == 'kentucky_river':
        baseline_elevation = 177
    else:
        raise Exception('Unknown place: {}'.format(args.place))

    ##########################################################################
    # Create output folders
    ##########################################################################
    os.makedirs(args.output_dir, exist_ok=True)

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
    with open(csv_path, 'r') as f:
        header = 0
        while True:
            line = f.readline()
            if 'image' in line:
                break
            header += 1
    alignment_data = pd.read_csv(csv_path, header=header)
    alignment_data = alignment_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    alignment_data.columns = alignment_data.columns.str.replace(' ', '')
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
    if not os.path.exists('temp.tiff'):
        with rasterio.open(label_raster_path) as label_src:
            labels = label_src.read()
            n_bands, height, width = labels.shape
            labels = labels.transpose(1, 2, 0)

            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(label_src.transform, rows, cols)

            profile = label_src.profile

            with rasterio.open(dsm_path) as dsm_src:
                dsm = dsm_src.read()
                dsm = medfilt2d(dsm.squeeze(), kernel_size=5)
                dsm_height, dsm_width = dsm.shape
                assert abs(height - dsm_height) <= 1 and abs(width - dsm_width) <= 1, \
                    'Label and DSM have different dimensions: {} vs {}, {} vs {}'.format(
                        height, dsm_height, width, dsm_width)
                dsm = cv2.resize(dsm, (width, height), interpolation=cv2.INTER_LINEAR)

                utm_z_grid = np.stack((xs, ys, dsm), axis=2)
                utm_z_labels_grid = np.concatenate((utm_z_grid, labels), axis=2)
                assert utm_z_labels_grid.shape == (height, width, 3 + n_bands)

        profile.update(
            dtype=rasterio.float32,
            count=3 + n_bands,
        )
        with rasterio.open('temp.tiff', 'w', **profile) as dst:
            dst.write(utm_z_labels_grid.transpose(2, 0, 1))
    else:
        with rasterio.open('temp.tiff', 'r') as src:
            utm_z_labels_grid = src.read().transpose(1, 2, 0)

    t1 = time.time()
    print('{:3f} seconds to read rasters'.format(t1 - t0))
    
    t1 = time.time()
    # Create mesh
    H, W, B = utm_z_labels_grid.shape
    print(utm_z_labels_grid.shape)
    xyz_labels = utm_z_labels_grid.reshape(H*W, B)
    prob_label = xyz_labels[:, 3:]

    xyz_gl = xyz_labels[:,:3]

    dt = startinpy.DT()
    dt.insert(xyz_gl, insertionstrategy="BBox")
    #-- exaggerate the elevation by a factor 2.0
    dt.write_ply("temp_surf.ply")

    # cloud = pv.PolyData(xyz_gl)
    # surf = cloud.delaunay_2d()
    
    t2 = time.time()
    print('{:3f} seconds to create mesh'.format(t2 - t1))
    
    # t1 = time.time()
    # surf.save('temp_surf.ply')  
    # t2 = time.time()
    # print('{:3f} seconds to save mesh'.format(t2 - t1))
    
    exit(0)
    crs = label_tiff_data.crs
    tform = pyproj.Transformer.from_crs("epsg:4326", "epsg:{}".format(crs.to_epsg()))

    # Setup camera matrices
    H, W = (512, 640)
    corrected_I, roi = cv2.getOptimalNewCameraMatrix(I, D, (W, H), 0, (W, H))

    ##########################################################################
    # Begin segmentation projection here
    ##########################################################################
    print('Starting segmentation estimation')
    image_paths = sorted(glob.glob(os.path.join(args.data_path, 'images/thermal/*')))[2000::1000]
    for t, img_path in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):

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

        img = cv2.imread(img_path, -1)
        img = thermal2rgb(img)
        undistorted_image = cv2.undistort(img, I, D, None, newcameramtx)

        print("Distance to ground plane: {}".format(dist_to_ground_plane))
        # dist_to_ground_plane = -9
        z = height + dist_to_ground_plane
        r = Rotation.from_quat(cam_xyzw)
        yaw, pitch, roll = r.as_euler('ZYX', degrees=False)

        # Image saving
        original_img = cv2.rotate(undistorted_image, cv2.ROTATE_180)
        t1 = time.time()
        mgl_mask = get_mask_mgl(labeled_camera_pts.T, corrected_I, colorize_func=colorize_func)
        t2 = time.time()
        print('{:3f} seconds to render mask'.format(t2 - t1))
        name = os.path.basename(img_path).split('.')[0]
        # cv2.imwrite('original_output/{}_autoseg_refined.png'.format(name), cv2.cvtColor(refined_mask, cv2.COLOR_RGB2BGR))
        overlay = cv2.addWeighted(original_img, 0.7, mgl_mask, 0.3, 0)
        cv2.imwrite('{}/{}_autoseg_mgl.png'.format(args.output_dir, name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
