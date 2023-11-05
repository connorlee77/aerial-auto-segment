import sys
sys.path.append('../')

import os
import argparse
import time

import numpy as np
import cv2
import rasterio
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch
import torch.nn.functional as F

import pydensecrf.densecrf as dcrf
import optuna

from utils.draw import colorize_common_landcover_label
from refine import dense_crf, read_and_preprocess_data, create_input_features

def create_model_params(trial, args):
    theta_alpha = trial.suggest_int('theta_alpha', 10, 200)
    theta_beta = trial.suggest_int('theta_beta', 0, 200)
    theta_gamma = trial.suggest_int('theta_gamma', 0, 200)
    w1 = trial.suggest_int('w1', 0, 20)
    w2 = trial.suggest_int('w2', 0, 20)
    
    kernel = args.kernel
    inference_steps = args.inference_steps

    crf_params = dict(
        theta_alpha=theta_alpha,
        theta_beta=theta_beta,
        theta_gamma=theta_gamma,
        w1=w1,
        w2=w2,
        kernel=kernel,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
        inference_steps=inference_steps,
    )
    return crf_params

def load_data(args, dataset_name):
    data_dict = read_and_preprocess_data(args.base_dir, args.epsg, dataset_name, args.resolution, args.unary_src, args.feature_set)
    feature_img = create_input_features(data_dict, args.feature_set)
    unary_probabilities = data_dict['unary'][:,:,:-1]

    gt_label = data_dict['ground_truth'].squeeze().astype(np.int8)
    
    # No labels should be greater than index 6 in common landcover label
    gt_label[gt_label > 6] = -1
    print(np.unique(gt_label))
    print(unary_probabilities.shape)
    return unary_probabilities, feature_img, gt_label

def objective(trial):
    crf_params = create_model_params(trial, args)
    
    total_loss = 0
    for dataset_name in args.datasets:
        unary_probabilities, feature_img, gt_label = load_data(args, dataset_name)
        estimated_labels, estimated_probabilities = dense_crf(unary_probabilities, feature_img, **crf_params)

        # Compute mIoU loss using pytorch gpu
        device = torch.device('cpu')
        torch_prob = torch.from_numpy(estimated_probabilities).unsqueeze(0).to(device)
        log_probs = torch.log(torch_prob)
        gt_label = torch.from_numpy(gt_label).unsqueeze(0).to(device)
        loss = F.nll_loss(log_probs.float(), gt_label, ignore_index=-1, reduction='mean').item()

        total_loss += loss.item()

    return total_loss / len(args.datasets)

def train(args):
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help='Base directory for data, i.e. "/data/chesapeake_bay_lulc/outputs/preprocessed"')
    parser.add_argument('--epsg', type=str, help='EPSG code for data, i.e "epsg-32618"')
    parser.add_argument('--datasets', nargs='+', type=str, help='List of dataset names in chesapeake bay dataset to use')
    parser.add_argument('--resolution', type=str, help='Resolution of data', choices=['0.6', '1.0', '2.0', 'original_resolution'])

    # Unary label source
    # TODO: add chesapeake bay-trained network as source
    parser.add_argument('--unary_src', type=str, help='Source of unary label', choices=['dynamicworld'])

    # Features to use for CRF
    parser.add_argument('--feature_set', nargs='+', help='List of features to use (at least one is required)', choices=['naip', 'planet', 'dem_1m', 'dem', 'dsm'], required=True)

    # CRF hyperparameters
    parser.add_argument('--kernel', type=str, help='Kernel type', choices=['full', 'diag', 'const'], default='diag')
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=5)
    
    # Optuna hyperparameters
    parser.add_argument('--n_trials', type=int, help='Number of trials', default=100)

    args = parser.parse_args()
    print(args)


    crf_params = dict(
        theta_alpha=5,
        theta_beta=5,
        theta_gamma=5,
        w1=5,
        w2=5,
        kernel='diag',
        normalization=dcrf.NORMALIZE_SYMMETRIC,
        inference_steps=1,
    )
    
    total_loss = 0
    for dataset_name in args.datasets:
        print(dataset_name)
        t1 = time.time()
        unary_probabilities, feature_img, gt_label = load_data(args, dataset_name)
        t2 = time.time()
        print(f'Load time: {t2-t1}')
        estimated_labels, estimated_probabilities = dense_crf(unary_probabilities, feature_img, **crf_params)
        t3 = time.time()
        print('CRF time: ', t3-t2)
        # Compute mIoU loss using pytorch gpu
        device = torch.device('cuda:0')
        torch_prob = torch.from_numpy(estimated_probabilities).unsqueeze(0).to(device)
        print(torch_prob.shape)
        a = torch_prob.sum(dim=0)
        print(a)
        print(torch.sum(a > 1))

        log_probs = torch.log(torch_prob)
        gt_label = torch.from_numpy(gt_label).unsqueeze(0).to(device)
        loss = F.nll_loss(log_probs.float(), gt_label.long(), ignore_index=-1, reduction='mean').item()
        t4 = time.time()
        print(f'Loss time: {t4-t3}')
        print(loss)
        total_loss += loss
        exit(0)
    # Visualization
    # colorized_unary_labels = colorize_common_landcover_label(data_dict['unary'][:,:,-1])
    # colorized_estimated_labels = colorize_common_landcover_label(estimated_labels)
    # colorized_gt_label = colorize_common_landcover_label(gt_label)

    # overlay = cv2.addWeighted(data_dict['naip'][:,:,:-1], 0.6, colorized_estimated_labels, 0.4, 0)
    # sxs1 = np.hstack([overlay, colorized_unary_labels])
    # sxs2 = np.hstack([colorized_estimated_labels, colorized_gt_label])
    # sxs = np.vstack([sxs1, sxs2])
    # cv2.imwrite('crf_inference.png', cv2.cvtColor(sxs, cv2.COLOR_RGB2BGR))
