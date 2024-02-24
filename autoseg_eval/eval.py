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

def deal_with_dontcare_class_in_refined_mask(refined_mask, dontcare_idx=255):
    if dontcare_idx in refined_mask:
        binary_mask = refined_mask == dontcare_idx
        # dilate mask then subtract the original mask to get the border
        border = cv2.dilate(binary_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) - binary_mask
        surrounding_classes = refined_mask[border == 1]
        most_freq_class, _ = scipy.stats.mode(surrounding_classes, axis=None, keepdims=False)
        refined_mask[binary_mask] = most_freq_class
        print('Set dontcare class to surrounding class: {}'.format(most_freq_class))

# not use f strings
def verify_args(args):
    assert os.path.exists(args.mask_dir), "Mask directory {} does not exist".format(args.mask_dir)
    assert os.path.exists(args.gt_dir), "GT directory {} does not exist".format(args.gt_dir)


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Colorize mask')
    parser.add_argument('--mask_dir', type=str, help='Path to the mask directory')
    parser.add_argument('--gt_dir', type=str, help='Path to the base image')

    parser.add_argument('--common_type', type=str, default='common',
                        choices=['common', 'more_common', 'most_common'], help='Type of common color map')
    parser.add_argument('--commonize-refined', action='store_true', help='Commonize the refined masks as well')

    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()

    # Get the mask files
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.png')))
    assert len(mask_files) > 0, 'No mask files found in {}'.format(args.mask_dir)

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
    # os.makedirs(args.output_dir, exist_ok=True)

    num_classes = max(new_name2idx.values()) + 1
    print(f'Number of classes: {num_classes}')
    cm = ConfusionMatrix(num_classes=num_classes)
    iou = IoU(cm)
    # Loop through the mask files
    for mask_file in tqdm.tqdm(mask_files):
        gt_file = os.path.join(args.gt_dir, os.path.basename(mask_file).replace('thermal-', 'pair-'))
        # Read the mask
        refined_mask = cv2.imread(mask_file, -1)
        gt_mask = cv2.imread(gt_file, -1)
        H, W = gt_mask.shape

        # Ignore the unknown (0) and background (1) classes
        gt_mask = np.clip(gt_mask.astype(np.int16) - 2, -1, None)
        # Commonize the GT mask. The refined mask is already commonized
        gt_mask_common = commonize_gt_labels(gt_mask, new_gt_idx2idx_map)

        if args.commonize_refined:
            # this is for ov-seg only.
            refined_mask = np.clip(refined_mask.astype(np.int16) - 2, 0, None)
            refined_mask = commonize_gt_labels(refined_mask, new_gt_idx2idx_map)

        # print(np.unique(refined_mask), num_classes, np.sum(refined_mask == num_classes), mask_file)

        # 255 is a don't care class: it can either be aberdeen proving ground from chesapeake bay or ice from dynamicworld.
        # They are both highly uncommon and should not appear in the prediction. Set them as the surrounding class as a way to deal with them.
        deal_with_dontcare_class_in_refined_mask(refined_mask, dontcare_idx=255)
        # deal_with_dontcare_class_in_refined_mask(refined_mask, dontcare_idx=num_classes)
        
        refined_mask_tensor = torch.from_numpy(refined_mask).long().view(1, H, W)
        refined_mask_tensor = F.one_hot(refined_mask_tensor, num_classes=num_classes).permute(0, 3, 1, 2).float()

        gt_mask_tensor = torch.from_numpy(gt_mask_common).long().view(1, H, W)
        iou.update((refined_mask_tensor, gt_mask_tensor))

    class_iou = iou.compute()
    miou = class_iou.mean().item()

    reverse_name2idx = {}
    for name, idx in new_name2idx.items():
        if idx in reverse_name2idx:
            reverse_name2idx[idx].append(name)
        else:
            reverse_name2idx[idx] = [name]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for idx, name_list in reverse_name2idx.items():
            if idx != -1:
                print('{}: {}'.format(', '.join(sorted(name_list)), class_iou[idx]), file=f)

        print("mIoU: {}".format(miou), file=f)
    confusion_matrix = cm.compute().numpy()
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), confusion_matrix)
