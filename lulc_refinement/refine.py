import sys

sys.path.append('../')

import argparse
import logging
import os
import time

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
import rasterio
from pydensecrf.utils import (create_pairwise_bilateral,
                              create_pairwise_gaussian)

from utils.draw import colorize_common_landcover_label

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def force_same_shape(data_dict):
    '''
        Force all geotiff data in the dictionary to have the same shape as the unary label

        ## Parameters: 
            data_dict: dictionary of data
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


def dense_crf(unary_potential, input_features, **params):
    '''
        Dense Conditional Random Field inference on a single image

        ## Parameters:
            unary_potential: 2D probability map (H, W, C) to be used as unary potentials in the CRF
            input_features: base image-like features (H, W, C) to be used for the pairwise potentials in the CRF
            params: dictionary of hyperparameters for the CRF
        ## Returns:
            map: CRF MAP output (H, W)
    '''

    if params['kernel'] == 'full':
        kernel = dcrf.FULL_KERNEL
    elif params['kernel'] == 'diag':
        kernel = dcrf.DIAG_KERNEL
    elif params['kernel'] == 'const':
        kernel = dcrf.CONST_KERNEL

    H, W, C = unary_potential.shape
    dcrf_probs = np.ascontiguousarray(unary_potential.transpose(2, 0, 1))
    U = -np.log(dcrf_probs + 1e-3)
    d = dcrf.DenseCRF2D(W, H, C)  # width, height, nlabels
    U = U.reshape(C, -1)

    d.setUnaryEnergy(U)

    _, _, n_channels = input_features.shape

    # If there is only one theta_beta value, use it for all channels (by passing a numerical type instead of a list)
    theta_betas_list = params['theta_betas']
    theta_betas = theta_betas_list[0] if len(theta_betas_list) == 1 else theta_betas_list

    pairwise_bilateral_energy = create_pairwise_bilateral(
        sdims=(params['theta_alpha'], params['theta_alpha']),
        schan=theta_betas,
        img=input_features,
        chdim=2,
    )

    pairwise_gaussian_energy = create_pairwise_gaussian(
        sdims=(params['theta_gamma'], params['theta_gamma']),
        shape=(H, W),
    )

    d.addPairwiseEnergy(pairwise_bilateral_energy,
                        compat=params['w1'], kernel=kernel, normalization=params['normalization'])
    d.addPairwiseEnergy(pairwise_gaussian_energy, compat=params['w2'],
                        kernel=kernel, normalization=params['normalization'])

    # TODO: Fix this hack. Should be simple removal of first if statement, but test it out.
    # if n_channels == 3:
    #     # Smoothness kernel
    #     d.addPairwiseGaussian(sxy=params['theta_gamma'], compat=params['w2'], kernel=kernel,
    #                         normalization=params['normalization'])
    #     # Appearance kernel
    #     d.addPairwiseBilateral(
    #         sxy=params['theta_alpha'],
    #         srgb=params['theta_beta'],
    #         rgbim=input_features,
    #         compat=params['w1'],
    #         kernel=kernel,
    #         normalization=params['normalization'],
    #     )
    # else:
    #     assert len(params['theta_beta_list']) == n_channels, 'theta_beta_list must have same number of channels as input features'
    #     pairwise_bilateral_energy = create_pairwise_bilateral(
    #         sdims=(params['theta_alpha'], params['theta_alpha']),
    #         schan=params['theta_beta_list'],
    #         img=input_features,
    #         chdim=2,
    #     )

    #     pairwise_gaussian_energy = create_pairwise_gaussian(
    #         sdims=(params['theta_gamma'], params['theta_gamma']),
    #         shape=(H, W),
    #     )

    #     d.addPairwiseEnergy(pairwise_bilateral_energy,
    #                         compat=params['w1'], kernel=kernel, normalization=params['normalization'])
    #     d.addPairwiseEnergy(pairwise_gaussian_energy, compat=params['w2'],
    #                         kernel=kernel, normalization=params['normalization'])

    Q = d.inference(params['inference_steps'])
    map = np.argmax(Q, axis=0).reshape(H, W)
    return map, np.asarray(Q).reshape(C, H, W)


def read_and_preprocess_data(base_dir, epsg, dataset, resolution, unary_src, feature_set):
    '''
        Read and preprocess geotiffs into np.array data for CRF refinement.

        ## Parameters:
            base_dir: base directory for data
            epsg: epsg code for data, i.e. epsg-XXXXX
            dataset: dataset name
            resolution: resolution of data
            unary_src: source of unary label (dynamic world, chesapeake bay-trained network)
            feature_set: list of features (naip, planet, dem_1m, dem, dsm) to use for CRF
        ## Returns:
            data_dict: dictionary of data to be used for CRF inference
    '''
    dataset_path = os.path.join(base_dir, epsg, dataset)
    assert os.path.exists(dataset_path), 'Dataset path {} does not exist'.format(dataset_path)

    # Ground truth label path
    chesapeake_bay_lulc_path = os.path.join(
        base_dir, epsg, dataset, 'chesapeake_bay_lc', resolution, 'converted_mosaic.tiff')

    # Unary label (probability) path
    dynamicworld_label_path = os.path.join(base_dir, epsg, dataset, 'dynamicworld', resolution, 'converted_mosaic.tiff')

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

    if os.path.exists(dynamicworld_label_path) and unary_src == 'dynamicworld':
        with rasterio.open(dynamicworld_label_path) as dynamicworld_label:
            data_dict['unary'] = dynamicworld_label.read()
            logging.info('dynamicworld nodata value: {}'.format(dynamicworld_label.nodata))

    # TODO: Add unary src for chesapeake bay-trained network

    # Image (features)
    for feature_name in feature_path_dict:
        feature_path = feature_path_dict[feature_name]
        if os.path.exists(feature_path):
            with rasterio.open(feature_path) as data:
                data_dict[feature_name] = data.read()
                logging.info('{} nodata value: {}'.format(feature_name, data.nodata))
        else:
            logging.warning('Feature "{}" not found at path {}'.format(feature_name, feature_path))

    assert 'unary' in data_dict, 'Unary label not found; check path {} and unary src {}'.format(
        dynamicworld_label_path, unary_src)
    # Rasters may have been off by 1 pixel in shape so make them the same as unary label
    force_same_shape(data_dict)

    return data_dict


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
        # TODO: add naip-derived features like NDVI, NDWI, etc.
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
    if 'dem' in feature_set:  # 10m resolution
        feature_img_list.append(preprocess_dxm(data_dict['dem']))
    if 'dsm' in feature_set:  # 2m resolution
        feature_img_list.append(preprocess_dxm(data_dict['dsm']))

    if 'surface_height' in feature_set:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        help='Base directory for data, i.e. "/data/chesapeake_bay_lulc/outputs/preprocessed"')
    parser.add_argument('--epsg', type=str, help='EPSG code for data, i.e "epsg-32618"')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--resolution', type=str, help='Resolution of data',
                        choices=['0.6', '1.0', '2.0', 'original_resolution'])

    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for CRF inference results')

    # Unary label source
    # TODO: add chesapeake bay-trained network as source
    parser.add_argument('--unary_src', type=str, help='Source of unary label', choices=['dynamicworld'])

    # Features to use for CRF
    parser.add_argument('--feature_set', nargs='+', help='List of features to use (at least one is required)',
                        choices=['naip', 'naip-nir', 'naip-ndvi', 'planet', 'dem_1m', 'dem', 'dsm', 'surface_height'], required=True)

    # CRF hyperparameters
    parser.add_argument('--theta_alpha', type=int, help='Appearance kernel (position) parameter', default=160)
    parser.add_argument('--theta_betas', type=int, nargs='+', help='Appearance kernel (color) parameters', default=[3])
    parser.add_argument('--theta_gamma', type=int, help='Smoothness kernel parameter', default=3)
    parser.add_argument('--w1', type=int, help='Appearance kernel weight', default=5)
    parser.add_argument('--w2', type=int, help='Smoothness kernel weight', default=3)
    parser.add_argument('--kernel', type=str, help='Kernel type', choices=['full', 'diag', 'const'], default='diag')
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=5)

    args = parser.parse_args()
    print(args)

    data_dict = read_and_preprocess_data(args.base_dir, args.epsg, args.dataset,
                                         args.resolution, args.unary_src, args.feature_set)
    feature_img = create_input_features(data_dict, args.feature_set)
    unary_probabilities = data_dict['unary'][:, :, :-1]

    crf_params = dict(
        theta_alpha=args.theta_alpha,
        theta_betas=args.theta_betas,
        theta_gamma=args.theta_gamma,
        w1=args.w1,
        w2=args.w2,
        kernel=args.kernel,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
        inference_steps=args.inference_steps,
    )

    estimated_labels, _ = dense_crf(unary_probabilities, feature_img, **crf_params)

    gt_label = data_dict['ground_truth'].squeeze()

    # Visualization
    colorized_unary_labels = colorize_common_landcover_label(data_dict['unary'][:, :, -1])
    colorized_estimated_labels = colorize_common_landcover_label(estimated_labels)
    colorized_gt_label = colorize_common_landcover_label(gt_label)

    overlay = cv2.addWeighted(data_dict['naip'][:, :, :-1], 0.6, colorized_estimated_labels, 0.4, 0)
    sxs1 = np.hstack([overlay, colorized_unary_labels])
    sxs2 = np.hstack([colorized_estimated_labels, colorized_gt_label])
    sxs = np.vstack([sxs1, sxs2])
    sxs = cv2.resize(sxs, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, 'sxs.png'), cv2.cvtColor(sxs, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, 'refined_labels.png'),
                cv2.cvtColor(colorized_estimated_labels, cv2.COLOR_RGB2BGR))
