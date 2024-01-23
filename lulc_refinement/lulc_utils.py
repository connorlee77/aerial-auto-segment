import logging
import os

import cv2
import numpy as np
import rasterio

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

feature_set_3D = set(['dem_1m', 'dem', 'dsm', 'surface_height'])


def check_using_3D_features(feature_set):
    for feature_name in feature_set:
        if feature_name in feature_set_3D:
            return True
    return False

####################################################################################################
# Helper functions for data I/O ###
####################################################################################################


def force_same_shape(data_dict):
    '''
        Force all geotiff data in the dictionary to have the same shape as the unary label

        ## Parameters: 
            data_dict: dictionary of np.array data
        ## Returns:
            None (Modify dict in-place)
    '''
    # Force everything to match the unary label shape
    C0, H0, W0 = data_dict['unary'].shape

    for key in data_dict:
        data = data_dict[key]
        C, H, W = data.shape

        # Check if the current data is close to the same shape as the unary label
        assert abs(H - H0) < 2 and abs(W - W0) < 2, \
            "{} ({},{}) and unary ({},{}) are too different in size".format(key, H, W, H0, W0)

        # Transpose to OpenCV H, W, C format
        data = data.transpose(1, 2, 0)
        if key == 'unary':
            pass
        elif key == 'ground_truth':
            data = cv2.resize(data, (W0, H0), interpolation=cv2.INTER_NEAREST)
        else:
            data = cv2.resize(data, (W0, H0), interpolation=cv2.INTER_LINEAR)

        # Overwrite original data and make sure single channel data is not squeezed
        data_dict[key] = data.reshape(H0, W0, C)


def read_and_preprocess_data(base_dir, epsg, dataset, resolution, unary_src, feature_set, unary_filename='converted_mosaic.tiff'):
    '''
        Read and preprocess geotiffs into np.array data for CRF refinement.

        ## Parameters:
            base_dir: base directory for data
            epsg: epsg code for data, i.e. epsg-XXXXX
            dataset: dataset name
            resolution: resolution of data
            unary_src: source of unary label (dynamic world, chesapeake bay-trained network, open earth map)
            feature_set: list of features (naip, planet, dem_1m, dem, dsm) to use for CRF
        ## Returns:
            data_dict: dictionary of data to be used for CRF inference
    '''
    dataset_path = os.path.join(base_dir, epsg, dataset)
    assert os.path.exists(dataset_path), 'Dataset path {} does not exist'.format(dataset_path)

    # Ground truth label path
    chesapeake_bay_lulc_path = os.path.join(
        base_dir, epsg, dataset, 'chesapeake_bay_lc', resolution, unary_filename)
    # Unary label (probability) path
    unary_label_path = os.path.join(base_dir, epsg, dataset, unary_src, resolution, unary_filename)

    # Input image (features) paths
    naip_path = os.path.join(base_dir, epsg, dataset, 'naip', resolution, 'mosaic.tiff')
    planet_path = os.path.join(base_dir, epsg, dataset, 'planet', resolution, 'mosaic.tiff')
    dem_1m_path = os.path.join(base_dir, epsg, dataset, 'dem_1m', resolution, 'mosaic.tiff')
    dem_path = os.path.join(base_dir, epsg, dataset, 'dem', resolution, 'mosaic.tiff')
    dsm_path = os.path.join(base_dir, epsg, dataset, 'dsm', resolution, 'mosaic.tiff')

    feature_path_dict = {
        'naip': naip_path,
        'planet': planet_path,
        'dem_1m': dem_1m_path,
        'dem': dem_path,
        'dsm': dsm_path,
    }

    # Read geotiff data into memory. Throw out transfomation data.
    data_dict = {}

    if os.path.exists(chesapeake_bay_lulc_path):
        with rasterio.open(chesapeake_bay_lulc_path) as chesapeake_bay_lc_label:
            data_dict['ground_truth'] = chesapeake_bay_lc_label.read()
            logging.info('chesapeake_bay_lc nodata value: {}'.format(chesapeake_bay_lc_label.nodata))

    if os.path.exists(unary_label_path):
        with rasterio.open(unary_label_path) as unary:
            data_dict['unary'] = unary.read()
            logging.info('Unary nodata value: {}'.format(unary.nodata))
    else:
        raise Exception('Unary label not found at path {} for unary source {}'.format(unary_label_path, unary_src))

    # Image (features)
    for feature_name in feature_path_dict:
        feature_path = feature_path_dict[feature_name]
        if os.path.exists(feature_path):
            with rasterio.open(feature_path) as data:
                data_dict[feature_name] = data.read()
                logging.info('{} nodata value: {}'.format(feature_name, data.nodata))
        else:
            logging.warning('Feature "{}" not found at path {}'.format(feature_name, feature_path))

    # Rasters may have been off by 1 pixel in shape so make them the same as unary label
    force_same_shape(data_dict)

    return data_dict


def create_input_features(data_dict, feature_set):
    '''
        Combine different input features in preparation for CRF refinement

        ## Parameters:
            data_dict: dictionary of data to be used for CRF inference
            feature_set: list of image features (naip, planet, dem_1m, dem, dsm) to use for CRF
        ## Returns:
            feature_img: input features (H, W, C) for CRF refinement
    '''
    derived_feature_set = set(['naip-nir', 'naip-ndvi', 'surface_height'])
    for feat_name in feature_set:
        if feat_name in derived_feature_set:
            continue
        assert feat_name in data_dict, 'Input feature {} not found in data_dict'.format(feat_name)

    feature_img_list = []
    if 'naip' in feature_set:  # 1m resolution
        feature_img_list.append(data_dict['naip'][:, :, :-1])

    if 'naip-nir' in feature_set:  # 1m resolution
        feature_img_list.append(data_dict['naip'][:, :, 3:4])

    if 'naip-ndvi' in feature_set:  # 1m resolution
        ndvi = compute_ndvi(data_dict['naip'])
        feature_img_list.append(ndvi)

    if 'planet' in feature_set:  # 3m resolution
        feature_img_list.append(data_dict['planet'])

    if 'dem_1m' in feature_set:  # 1m resolution
        feature_img_list.append(preprocess_dxm(data_dict['dem_1m']))
    elif 'dem' in feature_set:  # 10m resolution
        feature_img_list.append(preprocess_dxm(data_dict['dem']))
    elif 'dsm' in feature_set:  # 2m resolution
        feature_img_list.append(preprocess_dxm(data_dict['dsm']))
    elif 'surface_height' in feature_set:
        assert 'dsm' in data_dict, 'DSM not found in data_dict for surface height computation'

        if 'dem_1m' in data_dict:
            surface_height = compute_surface_height(data_dict['dsm'], data_dict['dem_1m'])
        elif 'dem' in data_dict:
            surface_height = compute_surface_height(data_dict['dsm'], data_dict['dem'])
        else:
            raise ValueError('No DEMs available in data_dict for surface height computation')

        feature_img_list.append(surface_height)

    feature_img = np.concatenate(feature_img_list, axis=2)
    logging.info('Input features has shape {}'.format(feature_img.shape))
    return feature_img

def label_and_probability_to_geotiff(label, probability, unary_geotiff_path, output_geotiff_path):
    '''
        Saves a np.array label to geotiff

        ## Parameters:
            label: np.array, (H, W)
            probability: np.array, (C, H, W)
            unary_geotiff_path: path to unary geotiff
            output_geotiff_path: path to output geotiff
        ## Returns:
            None
    '''

    
    assert probability.shape[1:] == label.shape, 'probability shape {} does not match label shape {}'.format(probability.shape, label.shape)
    assert len(label.shape) == 2, 'label must be 2D, receieved shape {}'.format(label.shape)
    
    bands = probability.shape[0] + 1
    data = np.concatenate([probability, label[np.newaxis, :, :]], axis=0)
    with rasterio.open(unary_geotiff_path, 'r') as src:
        profile = src.profile
        profile.update(
            count=bands,
            dtype=rasterio.float32,
            nodata=-9999,
        )

        with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32))

def label_to_geotiff(label, unary_geotiff_path, output_geotiff_path):
    '''
        Saves a np.array label to geotiff

        ## Parameters:
            label: np.array, (H, W)
            unary_geotiff_path: path to unary geotiff
            output_geotiff_path: path to output geotiff
        ## Returns:
            None
    '''

    assert len(label.shape) == 2, 'label must be 2D, receieved shape {}'.format(label.shape)
    with rasterio.open(unary_geotiff_path, 'r') as src:
        profile = src.profile
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=255,
        )

        with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
            dst.write(label.astype(rasterio.uint8), 1)

####################################################################################################
# Helper functions for preprocessing data ###
####################################################################################################


def compute_ndvi(naip_img):
    '''
        Compute NDVI from NAIP image

        ## Parameters:
            naip_img: NAIP image (H, W, 4)
        ## Returns:
            ndvi: NDVI image (H, W, 1)
    '''
    H, W, C = naip_img.shape
    nir_band = naip_img[:, :, 3]
    r_band = naip_img[:, :, 0]
    ndvi = (nir_band - r_band) / (nir_band + r_band + 1e-7)
    # HACK: Shift NDVI to [0, 2] to not break permutohedral lattice code. Actual value of NDVI does not matter for CRFs.
    ndvi = np.clip(ndvi + 1, 0, 2)
    return ndvi.reshape(H, W, 1)


def preprocess_dxm(dxm_img):
    '''
        Preprocess DEM/DSM image by removing outliers. Assume sea level is lowest point in dataset.

        ## Parameters:
            dxm_img: DEM/DSM image (H, W, 1)
        ## Returns:
            dxm_img: preprocessed DEM/DSM image (H, W, 1)
    '''
    # Fill nodata values with 0
    dxm_img[dxm_img <= -9000] = 0
    return np.clip(dxm_img, 0, np.percentile(dxm_img, 99.9))


def compute_surface_height(dsm_img, dem_img):
    '''
        Compute surface height from DSM and DEM images.

        ## Parameters:
            dsm_img: DSM image (H, W, 1)
            dem_img: DEM image (H, W, 1)
        ## Returns:
            surface_height: surface height image (H, W, 1)
    '''
    # Fill nodata values with 0
    dsm_img[dsm_img <= -9000] = 0
    dem_img[dem_img <= -9000] = 0

    diff = dsm_img - dem_img
    diff = np.clip(diff, 0, np.percentile(diff, 99.9))
    return diff
