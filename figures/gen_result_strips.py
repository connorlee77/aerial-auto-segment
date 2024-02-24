import sys
sys.path.append('..')
import os
import glob
import numpy as np
import cv2
import scipy
import tqdm
from autoseg_eval.utils import (
    get_color_map,
    colorize,
    get_converted_gt_mapping,
    cartd_idx2name,
    common_cartd_name2idx,
    more_common_cartd_name2idx,
    most_common_cartd_name2idx,
    commonize_gt_labels,
)

data = {
    # common index
    'dw-planet': '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/{}/open_sam_boxnms_0p50/dynamicworld/dem/1.0/crf_planet/{}/{}/{}',
    # common index
    'dw-none': '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/{}/open_sam_boxnms_0p50/dynamicworld/dem/1.0/none/{}/{}/{}',
    # common index
    'odise': '/home/connor/repos/aerial-auto-segment/autoseg_vfm/aerial_autoseg_outputs/{}/odise/{}/{}/mask/{}',

    # common index
    'oem-naip': '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/{}/open_sam_boxnms_0p50/open_earth_map_unet_lc_naip_corrected/dem/1.0/none/{}/{}/{}',

    # 10 index
    'gt': '/data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/{}/{}/masks/{}',
    'image': '/data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/{}/{}/thermal8/{}',
}

def deal_with_dontcare_class_in_refined_mask(refined_mask, dontcare_idx=255):
    if dontcare_idx in refined_mask:
        binary_mask = refined_mask == dontcare_idx
        # dilate mask then subtract the original mask to get the border
        border = cv2.dilate(binary_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) - binary_mask
        surrounding_classes = refined_mask[border == 1]
        most_freq_class, _ = scipy.stats.mode(surrounding_classes, axis=None, keepdims=False)
        refined_mask[binary_mask] = most_freq_class
        print('Set dontcare class to surrounding class: {}'.format(most_freq_class))

def get_image(data, label_set, place, trajectory, image_name):
    gt_path = data['gt'].format(place, trajectory, image_name.replace('thermal-', 'pair-'))
    image_path = data['image'].format(place, trajectory, image_name.replace('thermal-', 'pair-'))

    odise = data['odise'].format(label_set, place, trajectory, image_name.replace('thermal-', 'pair-'))
    dw = data['dw-none'].format(label_set, place, trajectory, image_name)
    dw_planet = data['dw-planet'].format(label_set, place, trajectory, image_name)
    oem = data['oem-naip'].format(label_set, place, trajectory, image_name)

    assert os.path.exists(gt_path), 'GT mask does not exist: {}'.format(gt_path)
    assert os.path.exists(image_path), 'Image does not exist: {}'.format(image_path)
    assert os.path.exists(odise), 'ODISE mask does not exist: {}'.format(odise)
    assert os.path.exists(dw), 'DW mask does not exist: {}'.format(dw)
    assert os.path.exists(dw_planet), 'DW Planet mask does not exist: {}'.format(dw_planet)
    assert os.path.exists(oem), 'OEM mask does not exist: {}'.format(oem)

    data = {
        'gt': cv2.imread(gt_path, -1),
        'image': cv2.imread(image_path, 1),
        'odise': cv2.imread(odise, -1),
        'dw-none': cv2.imread(dw, -1),
        'dw-planet': cv2.imread(dw_planet, -1),
        'oem-naip': cv2.imread(oem, -1),
    }

    # downsample images
    for key in data:
        interp = cv2.INTER_AREA
        if key != 'image':
            interp = cv2.INTER_NEAREST
        
        data[key] = cv2.resize(data[key], (320, 256), interpolation=interp)

    return data

def commonize_image_dict(image_dict, idx_map):
    for key in ['gt']:
        value = np.clip(image_dict[key].astype(np.int16) - 2, -1, None)
        deal_with_dontcare_class_in_refined_mask(value, dontcare_idx=255)
        image_dict[key] = commonize_gt_labels(value, idx_map)

    return image_dict

def colorize_image_dict(image_dict, label_set):
    colorized_image_dict = {}
    for key, value in image_dict.items():
        if key != 'image':
            colorized_image_dict[key] = colorize(value, image_dict['image'], label_set)[1]

    return colorized_image_dict

# This is only for converting the GT masks. The refined masks are already commonized
gt_common_idx2idx = get_converted_gt_mapping(cartd_idx2name, common_cartd_name2idx)
gt_more_common_idx2idx = get_converted_gt_mapping(cartd_idx2name, more_common_cartd_name2idx)
gt_most_common_idx2idx = get_converted_gt_mapping(cartd_idx2name, most_common_cartd_name2idx)

common_cmap = get_color_map('default')
more_common_cmap = get_color_map('more')
most_common_cmap = get_color_map('most')

output_folder = '/home/connor/repos/aerial-auto-segment/figures/results/test_strips'
os.makedirs(output_folder, exist_ok=True)

dw_planet = '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/*/open_sam_boxnms_0p50/dynamicworld/dem/1.0/crf_planet/*/*/*.png'
for path in tqdm.tqdm(glob.glob(dw_planet)[::-1]):
    split_path = path.split('/')
    label_set, place, trajectory, name = split_path[-9], split_path[-3], split_path[-2], split_path[-1]
    image_dict = get_image(data, label_set, place, trajectory, name)

    if label_set == 'common':
        common_image_dict = commonize_image_dict(image_dict, gt_common_idx2idx)
        colorized_image_dict = colorize_image_dict(common_image_dict, 'default')
    elif label_set == 'more_common':
        common_image_dict = commonize_image_dict(image_dict, gt_more_common_idx2idx)
        colorized_image_dict = colorize_image_dict(common_image_dict, 'more')
    elif label_set == 'most_common':
        common_image_dict = commonize_image_dict(image_dict, gt_most_common_idx2idx)
        colorized_image_dict = colorize_image_dict(common_image_dict, 'most')

    dw_planet_img = colorized_image_dict['dw-planet']
    dw_img = colorized_image_dict['dw-none']
    odise_img = colorized_image_dict['odise']
    oem_img = colorized_image_dict['oem-naip']
    gt_img = colorized_image_dict['gt']

    # stack them
    img = np.hstack([image_dict['image'], dw_planet_img, dw_img, odise_img, oem_img, gt_img])
    save_path = os.path.join(output_folder, label_set, place, trajectory)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, name), img)

    # save_path = os.path.join(output_folder, label_set, place, trajectory, name.split('.')[0])
    # os.makedirs(save_path, exist_ok=True)
    # cv2.imwrite(os.path.join(save_path, 'dw_planet.png'), colorized_image_dict['dw-planet'])
    # cv2.imwrite(os.path.join(save_path, 'dw_none.png'), colorized_image_dict['dw-none'])
    # cv2.imwrite(os.path.join(save_path, 'odise.png'), colorized_image_dict['odise'])
    # cv2.imwrite(os.path.join(save_path, 'oem_naip.png'), colorized_image_dict['oem-naip'])
    # cv2.imwrite(os.path.join(save_path, 'gt.png'), colorized_image_dict['gt'])
    # cv2.imwrite(os.path.join(save_path, 'image.png'), image_dict['image'])