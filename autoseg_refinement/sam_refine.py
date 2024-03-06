import os
import glob
import cv2
import numpy as np
import argparse
import tqdm
import scipy
from draw import colorize
import shutil


common_classes_name2idx = {
    'water': 0,
    'trees': 1,
    'low_vegetation': 2,
    'built': 3,
    'ground': 4,
    'sky': 5,
    'dont_care': -1,
}

# water, vegetation, built, ground, sky
more_common_classes_name2idx = {
    'water': 0,
    'trees': 1,
    'low_vegetation': 1,
    'built': 2,
    'ground': 3,
    'sky': 4,
    'dont_care': -1,
}

# water, land, sky
most_common_classes_name2idx = {
    'water': 0,
    'trees': 1,
    'low_vegetation': 1,
    'built': 1,
    'ground': 1,
    'sky': 2,
    'dont_care': -1,
}

dynamicworld_idx2name = {
    0: 'water',
    1: 'trees',
    2: 'grass',
    3: 'flooded_vegetation',
    4: 'crops',
    5: 'shrub_and_scrub',
    6: 'built',
    7: 'bare',
    8: 'snow_and_ice',
    255: 'sky',  # not in the original dynamic world classes
}

dynamicworld_old2new = {
    'water': 'water',
    'trees': 'trees',
    'grass': 'low_vegetation',
    'flooded_vegetation': 'low_vegetation',
    'crops': 'low_vegetation',
    'shrub_and_scrub': 'low_vegetation',
    'built': 'built',
    'bare': 'ground',
    'snow_and_ice': 'dont_care',
    'sky': 'sky',  # not in the original dynamic world classes
}

chesapeake_bay_idx2name = {
    0: 'water',
    1: 'tree_canopy_and_shrubs',
    2: 'low_vegetation',
    3: 'barren',
    4: 'impervious_surfaces',
    5: 'impervious_roads',
    6: 'aberdeen_proving_ground',
    255: 'sky',  # not in the original chesapeake bay classes
}

chesapeake_bay_old2new = {
    'water': 'water',
    'tree_canopy_and_shrubs': 'trees',
    'low_vegetation': 'low_vegetation',
    'barren': 'ground',
    'impervious_surfaces': 'built',
    'impervious_roads': 'built',
    'aberdeen_proving_ground': 'dont_care',
    'sky': 'sky',  # not in the original chesapeake bay classes
}

open_earth_map_idx2name = {
    0: 'Bareland',
    1: 'Grass', # rangeland
    2: 'Pavement', # developed
    3: 'Road', 
    4: 'Tree',
    5: 'Water',
    6: 'Cropland', # agriculture
    7: 'buildings',
    255: 'sky',  # not in the original open earth map classes
}

open_earth_map_old2new = {
    'Bareland': 'ground',
    'Grass': 'low_vegetation',
    'Pavement': 'built',
    'Road': 'built',
    'Tree': 'trees',
    'Water': 'water',
    'Cropland': 'low_vegetation',
    'buildings': 'built',

    'sky': 'sky',  # not in the original open earth map classes
}


def get_converted_mapping(old_idx2name, old_name2new_name, new_name2idx):
    new_class_set = set()
    new_class_idx_set = set()
    new_mapping = {}
    for k, v in old_idx2name.items():
        equivalent_class = old_name2new_name[v]
        new_idx = new_name2idx[equivalent_class]
        new_mapping[k] = new_idx

        new_class_set.add(equivalent_class)
        new_class_idx_set.add(new_idx)
        print('{} ({}) -> {} ({})'.format(v, k, equivalent_class, new_idx))

    print('New class set:', new_class_set)
    for i in range(0, max(new_name2idx.values()) + 1):
        assert i in new_class_idx_set, 'Class index {} not found in new class set'.format(i)

    return new_mapping


def commonize_labels(mask, class_mapping):
    new_mask = np.zeros_like(mask)
    for i in np.unique(mask):
        new_mask[mask == i] = class_mapping[i]
    return new_mask


def split_unlabeled_segments(mask):
    H, W = mask.shape
    split_mask = np.zeros((H, W), dtype=np.int32) - 1

    new_idx = 0
    for i in np.unique(mask):
        bin_mask = mask == i
        # before = bin_mask.copy()
        kernel = np.ones((11, 11), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        # os.makedirs('test123/masks', exist_ok=True)
        # cv2.imwrite('test123/masks/{}.png'.format(str(i).zfill(3)), np.hstack([bin_mask*255, before*255]).astype(np.uint8))
        contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(split_mask, [c], -1, new_idx, cv2.FILLED)
            new_idx += 1

    return split_mask


def refine_semantic_mask_with_sam(unrefined_semantic_mask, sam_mask):
    H, W = unrefined_semantic_mask.shape
    # Make index -1 an ignore class. This shouldn't be present in final mask.
    new_mask = np.zeros((H, W), dtype=np.int32) - 1

    # Index 0 in SAM masks are not segmented, with objects starting at index 1.
    # Semantic masks are 0 indexed, with 255 being the probable sky class.

    # Note: 255 is anything that is not segmented in the SAM mask. This can be independent segments.
    sam_mask = split_unlabeled_segments(sam_mask)

    for i in np.unique(sam_mask):
        index_mask = sam_mask == i
        most_freq_class, _ = scipy.stats.mode(unrefined_semantic_mask[index_mask], axis=None, keepdims=False)
        new_mask[index_mask] = most_freq_class

    assert -1 not in new_mask, 'Ignore class should not be present in final mask'

    return new_mask


def verify_args(args):
    print(args)
    assert os.path.exists(args.sam_predicted_mask_dir), 'Path {} does not exist'.format(args.sam_predicted_mask_dir)
    assert os.path.exists(args.unrefined_semantic_mask_dir), 'Path {} does not exist'.format(
        args.unrefined_semantic_mask_dir)
    assert os.path.exists(args.data_dir), 'Path {} does not exist'.format(args.data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refine SAM predicted masks')
    parser.add_argument('--sam_predicted_mask_dir', type=str, required=True,
                        help='Path to directory containing SAM predicted masks')
    parser.add_argument('--unrefined_semantic_mask_dir', type=str, required=True,
                        help='Path to directory containing unrefined semantic masks')
    parser.add_argument('--class_set', type=str, required=True, help='Class set to use for colorization',
                        choices=['dynamicworld', 'chesapeake', 'open_earth_map'])

    parser.add_argument('--commonize', action='store_true', help='Commonize the class set')
    parser.add_argument('--commonize_to', type=str, choices=['default', 'more', 'most'], default='default')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing thermal images and ground truth masks')
    parser.add_argument('--output_dir', default='outputs', type=str, help='Path to directory to save refined masks')
    args = parser.parse_args()
    verify_args(args)

    class_coloring = args.class_set

    common_name2idx_mapping = common_classes_name2idx
    if args.commonize_to == 'more':
        common_name2idx_mapping = more_common_classes_name2idx
    elif args.commonize_to == 'most':
        common_name2idx_mapping = most_common_classes_name2idx

    # Get common class mapping
    if args.class_set == 'dynamicworld':
        new_mapping = get_converted_mapping(dynamicworld_idx2name, dynamicworld_old2new, common_name2idx_mapping)
    elif args.class_set == 'chesapeake':
        new_mapping = get_converted_mapping(chesapeake_bay_idx2name, chesapeake_bay_old2new, common_name2idx_mapping)
    elif args.class_set == 'open_earth_map':
        new_mapping = get_converted_mapping(open_earth_map_idx2name, open_earth_map_old2new, common_name2idx_mapping)
    else:
        raise ValueError(f'No common class mapping found for {args.class_set}')

    os.makedirs(args.output_dir, exist_ok=True)
    unrefined_semantic_mask_paths = sorted(list(filter(lambda x: 'color' not in x, glob.glob(
        os.path.join(args.unrefined_semantic_mask_dir, '*_mask.png')))))
    # unrefined_semantic_mask_paths = sorted(glob.glob(
    #     os.path.join(args.unrefined_semantic_mask_dir, '*_mask.png')))

    for unrefined_semantic_mask_path in tqdm.tqdm(unrefined_semantic_mask_paths):
        # 18611, 26632 49073
        place, trajectory, name = unrefined_semantic_mask_path.split(os.path.sep)[-3:]
        basic_name = os.path.basename(unrefined_semantic_mask_path).replace('_mask.', '.')
        # if '32461' not in basic_name:
        #     continue
        sam_predicted_mask_path = os.path.join(args.sam_predicted_mask_dir, place, trajectory, basic_name)
        assert os.path.exists(sam_predicted_mask_path), f'No SAM predicted mask found for {sam_predicted_mask_path}'

        # Get corresponding original image
        cogito_dir = place
        if 'Duck' in place:
            cogito_dir = 'caltech_duck'
        elif 'Kentucky' in place:
            cogito_dir = 'kentucky_river'

        original_image_path = os.path.join(args.data_dir, cogito_dir, trajectory, 'thermal8',
                                           basic_name.replace('thermal-', 'pair-'))
        assert os.path.exists(original_image_path), f'No original image found for {original_image_path}'

        gt_mask_path = os.path.join(args.data_dir, cogito_dir, trajectory, 'masks',
                                    basic_name.replace('thermal-', 'pair-'))
        assert os.path.exists(gt_mask_path), f'No ground truth mask found for {gt_mask_path}'

        # Read images and masks
        unrefined_semantic_mask = cv2.imread(unrefined_semantic_mask_path, -1)
        sam_predicted_mask = cv2.imread(sam_predicted_mask_path, -1)
        original_image = cv2.imread(original_image_path, 1)
        # gt_mask = cv2.imread(gt_mask_path, -1)

        # Map the semantics into common classes
        if args.commonize:
            unrefined_semantic_mask = commonize_labels(unrefined_semantic_mask, new_mapping)
            class_coloring = 'common'
        # Refine the mask
        refined_mask = refine_semantic_mask_with_sam(unrefined_semantic_mask, sam_predicted_mask)
        
        # Postprocess for saving
        colorized_mask, overlay_img = colorize(refined_mask, class_coloring,
                                               base_image=original_image, common_type=args.commonize_to)
        save_path = os.path.join(args.output_dir, basic_name)

        cv2.imwrite(save_path, refined_mask)
        # cv2.imwrite(save_path.replace('refined.png', 'refined-color.png'), colorized_mask)
        # if overlay_img is not None:
        #     cv2.imwrite(save_path.replace('refined.png', 'refined-overlay.png'), overlay_img)

        # shutil.copy(sam_predicted_mask_path.replace('cartd_labeled_sam_png_masks', 'cartd_labeled_sam_masks_overlay'),
        #             os.path.join(args.output_dir, basic_name.replace('.png', '_sam.png')))
        # shutil.copy(unrefined_semantic_mask_path.replace('mask', 'color_mask'),
        #             os.path.join(args.output_dir, basic_name.replace('.png', '_unrefined.png')))

        # Save an image strip at half size for each image.
        sam_predicted_mask = cv2.imread(sam_predicted_mask_path.replace(
            'cartd_labeled_sam_png_masks', 'cartd_labeled_sam_masks_overlay'), 1)
        unrefined_semantic_mask = cv2.imread(unrefined_semantic_mask_path.replace('mask', 'overlay'), 1)

        # resize all images by half
        H, W = original_image.shape[:2]
        sam_predicted_mask = cv2.resize(sam_predicted_mask, (int(W / 2), int(H / 2)))
        unrefined_semantic_mask = cv2.resize(unrefined_semantic_mask, (int(W / 2), int(H / 2)))
        original_image = cv2.resize(original_image, (int(W / 2), int(H / 2)))
        overlay_img = cv2.resize(overlay_img, (int(W / 2), int(H / 2)))
        # colorized_mask = cv2.resize(colorized_mask, (int(W/2), int(H/2)))

        img_strip = np.hstack([original_image, unrefined_semantic_mask, sam_predicted_mask, overlay_img])
        cv2.imwrite(os.path.join(args.output_dir, basic_name.replace('.png', '_strip.jpg')), img_strip)

        # cv2.imwrite(os.path.join(args.output_dir, basic_name.replace('.png', '_overlay.jpg')), overlay_img)
        # cv2.imwrite(os.path.join(args.output_dir, basic_name.replace('.png', '_colorized.jpg')), colorized_mask)
        # cv2.imwrite(os.path.join(args.output_dir, basic_name.replace('.png', '_original.jpg')), original_image)
        # cv2.imwrite(os.path.join(args.output_dir, basic_name.replace('.png', '_sam.jpg')), sam_predicted_mask)
        # cv2.imwrite(os.path.join(args.output_dir, basic_name.replace('.png', '_unrefined.jpg')), unrefined_semantic_mask)