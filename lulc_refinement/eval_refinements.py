import sys

sys.path.append('../')

import argparse
import glob
import os

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import tqdm
from ignite.metrics import confusion_matrix, mIoU
from boundary_loss import BoundaryMetric

label_mapping = {
    0: 'water',
    1: 'trees',
    2: 'low vegetation',
    3: 'scrub and shrub',
    4: 'flooded vegetation',
    5: 'built',
    6: 'bare',
    7: 'snow and ice',
}


def read_label_raster(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read()
        meta = src.meta

        if meta['count'] != 1:
            data = data[-1]

        # NOTE: Ignore snow/ice (none of that in chesapeake bay set) and nodata values.
        data = data.astype(int)
        data[data > 6] = -1
        data = np.clip(data, -1, None)

        return data.squeeze(), meta


def load_data(label_path, gt_path, device):
    refined_labels, refined_meta = read_label_raster(label_path)
    gt_labels, gt_meta = read_label_raster(gt_path)
    # Sanity check; can check affine transform as well
    assert refined_meta['crs'] == gt_meta['crs'], 'CRS mismatch: {} vs {}'.format(
        refined_meta['crs'], gt_meta['crs'])

    H, W = refined_labels.shape

    # Make gt_labels same size as refined_labels
    gt_labels = cv2.resize(gt_labels, (W, H), interpolation=cv2.INTER_NEAREST)
    assert refined_labels.shape == gt_labels.shape, 'refined_labels shape {} != gt_labels shape {}'.format(
        refined_labels.shape, gt_labels.shape)

    refined_labels = torch.from_numpy(refined_labels).to(device).view(1, H, W)
    gt_labels = torch.from_numpy(gt_labels).to(device).view(1, H, W).long()

    # Fake the ignored class
    refined_labels[refined_labels == -1] = 7

    # one-hot vector of ground truth
    one_hot_refined_labels = F.one_hot(refined_labels, 8).float().permute(0, 3, 1, 2)
    one_hot_refined_labels = one_hot_refined_labels[:, :-1, :, :]  # slice away the ignored pixels
    return one_hot_refined_labels, gt_labels


def refined_predictions_generator(args, device):
    refined_files = glob.glob(os.path.join(args.refined_label_data_dir, 'epsg-*', '*', 'refined_lulc',
                              args.resolution, args.unary_src, args.base_img_src, 'refined_labels.tiff'))
    for refined_label_path in tqdm.tqdm(refined_files):
        epsg, place = refined_label_path.split('/')[-7:-5]
        ground_truth_path = os.path.join(args.ground_truth_data_dir, 'outputs', 'preprocessed',
                                         epsg, place, 'chesapeake_bay_lc', args.resolution, 'converted_mosaic.tiff')
        assert os.path.exists(ground_truth_path), 'Ground truth path {} does not exist'.format(ground_truth_path)

        refined_label, gt_label = load_data(refined_label_path, ground_truth_path, device)
        yield refined_label, gt_label


def dynamicworld_generator(args, device):
    dynamicworld_files = glob.glob(os.path.join(args.ground_truth_data_dir, 'outputs', 'preprocessed', 'epsg-*',
                                                '*', 'dynamicworld', args.resolution, 'converted_mosaic.tiff'))
    for dynamicworld_label_path in tqdm.tqdm(dynamicworld_files):
        ground_truth_path = dynamicworld_label_path.replace('dynamicworld', 'chesapeake_bay_lc')
        assert os.path.exists(ground_truth_path), 'Ground truth path {} does not exist'.format(ground_truth_path)
        dynamicworld_label, gt_label = load_data(dynamicworld_label_path, ground_truth_path, device)
        yield dynamicworld_label, gt_label


def eval(label_pair_generator, device):
    boundary_metric = BoundaryMetric(theta0=3, theta=5, ignore_index=-1)

    cm = confusion_matrix.ConfusionMatrix(num_classes=7, device=device)
    miou_metric = mIoU(cm)
    for refined_labels, gt_labels in label_pair_generator:
        miou_metric.update((refined_labels, gt_labels))
        boundary_metric.update((refined_labels, gt_labels))

    print('Boundary metric: ', boundary_metric.compute())
    print('mIoU: ', miou_metric.compute().item())
    cmatrix = cm.compute()
    intersection = torch.diag(cmatrix)
    union = cmatrix.sum(dim=0) + cmatrix.sum(dim=1) - torch.diag(cmatrix)
    iou = intersection / union
    print('--- Class IoUs ---')
    for i in range(len(iou)):
        print('{}: {}'.format(label_mapping[i], iou[i].item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refined-label-data-dir', type=str, default=None, help='Path to refined label data')
    parser.add_argument('--compute-dynamicworld-baseline', action='store_true',
                        help='Compute dynamicworld baseline. Labels should be stored in the same directory tree of --ground-truth-data-dir')
    parser.add_argument('--ground-truth-data-dir', type=str, required=True)

    # TODO: Add chesapeake bay-trained network as source
    parser.add_argument('--resolution', default='1.0', type=str, help='Resolution of data')
    parser.add_argument('--unary-src', default='dynamicworld', type=str,
                        choices=['dynamicworld'], help='Source of unary label')
    parser.add_argument('--base-img-src', default='naip', type=str,
                        help='Main image source used to refine CRF', choices=['naip', 'planet'])

    parser.add_argument('--device', type=int, help='GPU device number', default=None)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.device) if args.device is not None else 'cpu')

    if args.refined_label_data_dir is not None:
        print('Computing refined prediction metrics...')
        refined_predictions_generator = refined_predictions_generator(args, device)
        eval(refined_predictions_generator, device)
    
    if args.compute_dynamicworld_baseline:
        print('Computing dynamicworld baseline metrics...')
        dynamicworld_generator = dynamicworld_generator(args, device)
        eval(dynamicworld_generator, device)
