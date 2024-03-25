import os
import glob
import numpy as np
import cv2
import argparse
import tqdm
import torch
import torch.nn.functional as F
from ignite.metrics import IoU, ConfusionMatrix
import scipy
import time

from utils import (
    colorize,
    get_converted_gt_mapping,
    cartd_idx2name,
    common_cartd_name2idx,
    more_common_cartd_name2idx,
    most_common_cartd_name2idx,
    commonize_gt_labels,
)


# Mask: /home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/more_common/open_sam_boxnms_0p35/chesapeake_bay_swin_crossentropy_lc_planet/dem/1.0/crf_planet_surface_height/2021-09-09-KentuckyRiver/flight1-1/thermal-00180.png
# GT: /data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/kentucky_river/flight1-1/masks/pair-00000.png

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Colorize mask')
    parser.add_argument('--gt_dir', type=str, help='Path to the base image')

    parser.add_argument('--common_type', type=str, default='common',
                        choices=['common', 'more_common', 'most_common'], help='Type of common color map')

    parser.add_argument('--output_dir', type=str, default='cartd_cm', help='Path to the output directory')
    args = parser.parse_args()

    # Get the mask files
    mask_files = sorted(glob.glob(os.path.join(args.gt_dir, '*', '*', 'masks', '*.png')))
    assert len(mask_files) > 0, 'No mask files found in {}'.format(args.gt_dir)

    # This is only for converting the GT masks. The refined masks are already commonized
    if args.common_type == 'common':
        new_name2idx = common_cartd_name2idx
    elif args.common_type == 'more_common':
        new_name2idx = more_common_cartd_name2idx
    elif args.common_type == 'most_common':
        new_name2idx = most_common_cartd_name2idx
    else:
        raise ValueError('Unknown common type: {}'.format(args.common_type))

    new_gt_idx2idx_map = get_converted_gt_mapping(cartd_idx2name, new_name2idx)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    num_classes = max(new_name2idx.values()) + 1
    print(f'Number of classes: {num_classes}')
    # Loop through the mask files
    for gt_file in tqdm.tqdm(mask_files):
        place, trajectory, _, mask_name = gt_file.split(os.path.sep)[-4:]

        # Read the mask
        gt_mask = cv2.imread(gt_file, -1)
        H, W = gt_mask.shape

        # Ignore the unknown (0) and background (1) classes
        gt_mask = np.clip(gt_mask.astype(np.int16) - 2, -1, None)
        # Commonize the GT mask. The refined mask is already commonized
        gt_mask_common = commonize_gt_labels(gt_mask, new_gt_idx2idx_map)
        assert gt_mask_common.max() < num_classes, 'Max class index {} exceeds the number of classes {}'.format(
            gt_mask_common.max(), num_classes)

        gt_mask_common[gt_mask_common == -1] = 255
        assert gt_mask_common.min() >= 0, 'Min class index {} is less than 0'.format(gt_mask_common.min())

        save_dir = os.path.join(args.output_dir, place, trajectory, 'masks')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, mask_name)
        cv2.imwrite(save_path, gt_mask_common)

        if args.common_type == 'common':
            common_type = 'default'
        else:
            common_type = args.common_type.split('_')[0]
        colorized_mask = colorize(gt_mask_common, None, common_type)[0]
        save_path = os.path.join(save_dir, mask_name.replace('.png', '_colorized.jpg'))
        cv2.imwrite(save_path, colorized_mask)