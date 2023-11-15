import sys

sys.path.append('../')

import argparse
import logging
import os

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from dense_crf import dense_crf
from lulc_utils import (create_input_features, label_to_geotiff,
                        read_and_preprocess_data, check_using_3D_features)
from utils.draw import colorize_common_landcover_label

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


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
    parser.add_argument('--theta_alpha', type=float, help='Appearance kernel (position x, y) parameter', default=160)
    parser.add_argument('--theta_alpha_z', type=float, help='Appearance kernel (position z) parameter', default=-1)
    parser.add_argument('--theta_betas', type=float, nargs='+',
                        help='Appearance kernel (color) parameters', default=[3])
    parser.add_argument('--theta_gamma', type=float, help='Smoothness kernel parameter (x,y)', default=3)
    parser.add_argument('--theta_gamma_z', type=float, help='Smoothness kernel parameter (z)', default=-1)
    parser.add_argument('--w1', type=float, help='Appearance kernel weight', default=5)
    parser.add_argument('--w2', type=float, help='Smoothness kernel weight', default=3)
    parser.add_argument('--kernel', type=str, help='Kernel type', choices=['full', 'diag', 'const'], default='diag')
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=5)

    args = parser.parse_args()
    args.use_3d = check_using_3D_features(args.feature_set)
    print(args)

    data_dict = read_and_preprocess_data(args.base_dir, args.epsg, args.dataset,
                                         args.resolution, args.unary_src, args.feature_set)
    feature_img = create_input_features(data_dict, args.feature_set)
    unary_probabilities = data_dict['unary'][:, :, :-1]

    theta_alpha_z, theta_gamma_z = None, None
    if args.use_3d:
        assert args.theta_gamma_z > 0 and args.theta_alpha_z > 0, 'theta_gamma_z and theta_alpha_z must be positive'
        theta_alpha_z = args.theta_alpha_z
        theta_gamma_z = args.theta_gamma_z

    crf_params = dict(
        theta_alpha=args.theta_alpha,
        theta_alpha_z=theta_alpha_z,
        theta_betas=args.theta_betas,
        theta_gamma=args.theta_gamma,
        theta_gamma_z=theta_gamma_z,
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

    # Save refined labels to geotiff based on unary label's geotiff metadata. This is because we reshape all data to match the unary label's shape.
    unary_raster_path = os.path.join(args.base_dir, args.epsg, args.dataset,
                                     'dynamicworld', args.resolution, 'converted_mosaic.tiff')
    label_to_geotiff(
        estimated_labels,
        unary_raster_path,
        os.path.join(args.output_dir, 'refined_labels.tiff'),
    )
