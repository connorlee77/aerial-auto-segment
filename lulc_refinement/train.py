import sys

sys.path.append('../')

import argparse
import logging
import os

import cv2
import joblib
import numpy as np
import optuna
import pydensecrf.densecrf as dcrf
import torch
import torch.nn.functional as F
from boundary_loss import BoundaryLoss
from dense_crf import dense_crf
from lulc_utils import create_input_features, read_and_preprocess_data, check_using_3D_features
from utils.draw import colorize_common_landcover_label

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def create_model_params(trial, args):

    if args.constrained:
        # Constrained
        theta_alpha = trial.suggest_int('theta_alpha', 5, 200)  # 5, 200
        if not args.nonconstant_kernel_parameters:
            theta_betas_list = [trial.suggest_int('theta_beta', 1, 10)]  # 1, 10
        theta_gamma = trial.suggest_int('theta_gamma', 1, 10)  # 1, 10
        w1 = trial.suggest_int('w1', 1, 10)  # 1, 10
        w2 = trial.suggest_int('w2', 1, 10)  # 1, 10
    else:
        # Unconstrained
        theta_alpha = trial.suggest_float('theta_alpha', 0.1, 200)  # 5, 200
        if not args.nonconstant_kernel_parameters:
            theta_betas_list = [trial.suggest_float('theta_beta', 0.1, 200)]  # 1, 10
        theta_gamma = trial.suggest_float('theta_gamma', 0.1, 200)  # 1, 10
        w1 = trial.suggest_float('w1', 0.1, 50)  # 1, 10
        w2 = trial.suggest_float('w2', 0.1, 50)  # 1, 10

    if args.nonconstant_kernel_parameters:
        assert args.n_feature_channels > 0, 'Must specify number of feature channels if using nonconstant kernel parameters'
        theta_betas_list = []
        actual_channels = args.n_feature_channels
        if args.use_3d:
            actual_channels -= 1
        for i in range(actual_channels):
            theta_i = trial.suggest_float('theta_beta_{}'.format(i), 0.1, 200)
            theta_betas_list.append(theta_i)

    theta_alpha_z, theta_gamma_z = None, None
    if args.use_3d:
        theta_alpha_z = trial.suggest_float('theta_alpha_z', 0.1, 200)
        theta_gamma_z = trial.suggest_float('theta_gamma_z', 0.1, 200)

    kernel = args.kernel
    inference_steps = args.inference_steps

    crf_params = dict(
        theta_alpha=theta_alpha,
        theta_alpha_z=theta_alpha_z,
        theta_betas=theta_betas_list,
        theta_gamma=theta_gamma,
        theta_gamma_z=theta_gamma_z,
        w1=w1,
        w2=w2,
        kernel=kernel,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
        inference_steps=inference_steps,
    )
    return crf_params


def load_data(args, dataset_name):
    data_dict = read_and_preprocess_data(args.base_dir, args.epsg, dataset_name,
                                         args.resolution, args.unary_src, args.feature_set)
    feature_img = create_input_features(data_dict, args.feature_set)
    unary_probabilities = data_dict['unary'][:, :, :-1]

    gt_label = data_dict['ground_truth'].squeeze().astype(np.int8)

    # No labels should be greater than index 6 in common landcover label
    gt_label[gt_label > 6] = -1
    return unary_probabilities, feature_img, gt_label


def crf_inference_and_loss(unary_probabilities_tile, feature_img_tile, gt_label_tile, crf_params, args, trial_id, tile_counter):
    estimated_labels, estimated_probabilities = dense_crf(unary_probabilities_tile, feature_img_tile, **crf_params)

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.boundary_loss:
        device = torch.device('cuda:{}'.format(args.device_id))

    torch_prob = torch.from_numpy(estimated_probabilities).unsqueeze(0).to(device)
    gt_label_tile_tensor = torch.from_numpy(gt_label_tile).unsqueeze(0).to(device)

    if args.boundary_loss:

        boundary_loss = BoundaryLoss(
            theta0=args.boundary_loss_theta0, theta=args.boundary_loss_theta, ignore_index=-1)

        # Boundary loss HACK: Handle case where labels are uniform
        gt_label_classes = torch.unique(gt_label_tile_tensor)
        pred_label_classes = torch.unique(torch.argmax(torch_prob, dim=1))

        single_gt_pred = (len(gt_label_classes) == 1) and (len(pred_label_classes) == 1)
        if single_gt_pred and gt_label_classes[0] == pred_label_classes[0]:
            loss = 0  # Correct prediction
        elif single_gt_pred and gt_label_classes[0] != pred_label_classes[0]:
            loss = 1  # Very wrong prediction
        else:
            loss = boundary_loss(torch_prob.float(), gt_label_tile_tensor.long()).item()

            if args.augment_boundary_loss:
                log_probs = torch.log(torch_prob)
                loss += 0.7 * F.nll_loss(
                    log_probs.float(),
                    gt_label_tile_tensor.long(),
                    weight=None,
                    ignore_index=-1,
                    reduction='mean'
                ).item()
    else:
        weight = None
        if args.weight:
            # TODO: remove hardcoded weights for loss function
            weight = torch.tensor([1.00, 1.41, 2.85, 227.89, 4.78, 29.78, 28.16]).to(device)

        log_probs = torch.log(torch_prob)
        loss = F.nll_loss(
            log_probs.float(),
            gt_label_tile_tensor.long(),
            weight=weight,
            ignore_index=-1,
            reduction='mean'
        ).item()

    if args.visualize and trial_id % 10 == 0 and tile_counter % 4 == 0:
        unary_labels = np.argmax(unary_probabilities_tile, axis=2)
        colorized_unary_labels = colorize_common_landcover_label(unary_labels)
        colorized_estimated_labels = colorize_common_landcover_label(estimated_labels)
        colorized_gt_label = colorize_common_landcover_label(gt_label_tile)

        output_dir = os.path.join('outputs', args.study_name, 'train-tiles')
        os.makedirs(output_dir, exist_ok=True)

        # NOTE: Assume RGB image is always present
        sxs1 = np.hstack([feature_img_tile[:, :, 0:3].astype(np.uint8), colorized_unary_labels])
        sxs2 = np.hstack([colorized_estimated_labels, colorized_gt_label])
        sxs = np.vstack([sxs1, sxs2])
        sxs_small = cv2.resize(sxs, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)

        tile_name = 'trial_{}_tile_{}.png'.format(str(trial_id).zfill(4), str(tile_counter).zfill(3))
        cv2.imwrite(os.path.join(output_dir, tile_name), cv2.cvtColor(sxs_small, cv2.COLOR_RGB2BGR))

        # Give a warning if feature image has more than 3 channels for debugging purposes
        if feature_img_tile.shape[2] > 3:
            logging.info('Feature image has more than 3 channels. {} channels in use'.format(feature_img_tile.shape[2]))

    return loss


def crop_images(unary_probabilities, feature_img, gt_label, r1, r2, c1, c2):
    unary_probabilities_tile = np.ascontiguousarray(unary_probabilities[r1:r2, c1:c2, :])
    feature_img_tile = np.ascontiguousarray(feature_img[r1:r2, c1:c2, :])
    gt_label_tile = np.ascontiguousarray(gt_label[r1:r2, c1:c2])
    return unary_probabilities_tile, feature_img_tile, gt_label_tile


def load_training_data(args):
    tile_size = 2000
    overlap = int(tile_size / 4)

    data = []
    for dataset_name in args.datasets:
        unary_probabilities, feature_img, gt_label = load_data(args, dataset_name)
        H, W, C = unary_probabilities.shape

        r = 0
        last_row_hit = False
        while r + tile_size <= H:
            c = 0
            last_col_hit = False
            while c + tile_size <= W:
                unary_probabilities_tile, feature_img_tile, gt_label_tile = crop_images(
                    unary_probabilities,
                    feature_img,
                    gt_label,
                    r, r + tile_size, c, c + tile_size
                )

                # Add datasets if valid
                if np.sum(gt_label_tile >= 0) != 0:
                    data_entry = {
                        'unary_probabilities': unary_probabilities_tile,
                        'feature_img': feature_img_tile,
                        'gt_label': gt_label_tile,
                    }
                    data.append(data_entry)

                c += tile_size - overlap
                if not last_col_hit and c + tile_size > W:
                    c = W - tile_size
                    last_col_hit = True

            r += tile_size - overlap
            if not last_row_hit and r + tile_size > H:
                r = H - tile_size
                last_row_hit = True
    return data


def objective(trial):
    crf_params = create_model_params(trial, args)

    total_loss = 0
    for i, data_entry in enumerate(data):
        unary_probabilities_tile = data_entry['unary_probabilities']
        feature_img_tile = data_entry['feature_img']
        gt_label_tile = data_entry['gt_label']

        loss = crf_inference_and_loss(
            unary_probabilities_tile,
            feature_img_tile,
            gt_label_tile,
            crf_params,
            args,
            trial_id=trial.number,
            tile_counter=i,
        )

        total_loss += loss
    return total_loss / len(data)


def optimize_study(study_name, storage, objective, n_trials, mask):
    # NOTE: hack for parallel processing
    # os.system("taskset -p 0xff %d" % os.getpid())
    os.sched_setaffinity(os.getpid(), [mask])

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[optuna.study.MaxTrialsCallback(
            args.n_trials * args.parallel_jobs, states=(optuna.trial.TrialState.COMPLETE,))],
        gc_after_trial=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        help='Base directory for data, i.e. "/data/chesapeake_bay_lulc/outputs/preprocessed"')
    parser.add_argument('--epsg', type=str, help='EPSG code for data, i.e "epsg-32618"')
    parser.add_argument('--datasets', nargs='+', type=str,
                        help='List of dataset names in chesapeake bay dataset to use')
    parser.add_argument('--resolution', type=str, help='Resolution of data',
                        choices=['0.6', '1.0', '2.0', 'original_resolution'])

    # Unary label source
    # TODO: add chesapeake bay-trained network as source
    parser.add_argument('--unary_src', type=str, help='Source of unary label', choices=['dynamicworld'])

    # Features to use for CRF
    parser.add_argument('--feature_set', nargs='+', help='List of features to use (at least one is required)',
                        choices=['naip', 'naip-nir', 'naip-ndvi', 'planet', 'dem_1m', 'dem', 'dsm', 'surface_height'], required=True)

    # CRF hyperparameters
    parser.add_argument('--kernel', type=str, help='Kernel type', choices=['full', 'diag', 'const'], default='diag')
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=5)

    parser.add_argument('--nonconstant_kernel_parameters', action='store_true',
                        help='Use a different smoothing parameter for each image feature channel')
    parser.add_argument('--n_feature_channels', type=int, help='Number of channels in feature image', default=-1)

    # Optuna hyperparameters
    parser.add_argument('--study_name', type=str, help='Name of optuna study', default='chesapeake-bay-crf-tuning')
    parser.add_argument('--n_trials', type=int, help='Number of optuna trials', default=100)
    parser.add_argument('--parallel_jobs', type=int, help='Number of parallel jobs for optuna', default=1)
    parser.add_argument('--cores-to-use', type=int, nargs='+',
                        help='List of cores to use for parallel jobs', default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualize CRF inference')
    parser.add_argument('--constrained', action='store_true',
                        help='Use constrained CRF (theta parameters are integers)')

    # Loss function parameters
    parser.add_argument('--weight', action='store_true', help='Weight NLL loss function via label frequencies')
    parser.add_argument('--boundary_loss', action='store_true', help='Use boundary loss function')
    parser.add_argument('--boundary_loss_theta0', type=int, help='Theta0 for boundary loss function', default=3)
    parser.add_argument('--boundary_loss_theta', type=int, help='Theta for boundary loss function', default=5)
    parser.add_argument('--augment_boundary_loss', action='store_true',
                        help='Augment boundary loss with NLL loss function')
    parser.add_argument('--device_id', type=int, help='GPU device ID (needed to speed up boundary loss)', default=0)
    args = parser.parse_args()
    
    args.use_3d = check_using_3D_features(args.feature_set)
    print(args)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # args.study_name = "chesapeake-bay-crf-tuning"  # Unique identifier of the study.
    storage_name = "postgresql://connor:1898832@localhost/{}".format(args.study_name)
    # storage_name = "sqlite:///{}.db".format(args.study_name)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize")

    data_filename_memmap = '{}.dat'.format(args.study_name)
    if os.path.exists(data_filename_memmap):
        data = joblib.load(data_filename_memmap, mmap_mode='r+')
    else:
        data = load_training_data(args)
        joblib.dump(data, data_filename_memmap)
        data = joblib.load(data_filename_memmap, mmap_mode='r+')  # Load with mmap write permissions

    if args.parallel_jobs == 1:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[optuna.study.MaxTrialsCallback(args.n_trials, states=(optuna.trial.TrialState.COMPLETE,))],
            gc_after_trial=True)
    else:
        if args.cores_to_use is None:
            args.cores_to_use = list(range(args.parallel_jobs))
        else:
            assert len(args.cores_to_use) == args.parallel_jobs, 'Number of cores to use must match number of parallel jobs'
        joblib.Parallel(n_jobs=args.parallel_jobs, prefer='processes')(
            joblib.delayed(optimize_study)(
                args.study_name,
                storage_name,
                objective,
                args.n_trials,
                i) for i in args.cores_to_use
        )
