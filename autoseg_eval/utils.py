from PIL import ImageColor
import numpy as np
import cv2


def common_color_map(common_type='default'):
    common_colors = {}
    if common_type == 'default':
        common_colors = {
            0: '#419BDF',  # water
            1: '#397D49',  # trees (tree canopy)
            2: '#88B053',  # low vegetation (shrubs, grass, crops, flooded vegetation)
            3: '#C4281B',  # built (impervious structures, other impervious, impervious roads)
            4: '#ffffff',  # ground (bare, barren)
            5: '#01CCFA',  # sky
            -1: '#B39FE1',  # dont care (snow and ice, aberdeen proving ground)
        }
    elif common_type == 'more':
        common_colors = {
            0: '#419BDF',  # water
            1: '#397D49',  # vegetation
            2: '#C4281B',  # built (impervious structures, other impervious, impervious roads)
            3: '#ffffff',  # ground (bare, barren)
            4: '#01CCFA',  # sky
            -1: '#B39FE1',  # dont care (snow and ice, aberdeen proving ground)
        }
    elif common_type == 'most':
        common_colors = {
            0: '#419BDF',  # water
            1: '#ffffff',  # ground (bare, barren)
            2: '#01CCFA',  # sky
            -1: '#B39FE1',  # dont care (snow and ice, aberdeen proving ground)
        }

    color_map = {}
    for idx, color in common_colors.items():
        color_map[idx] = ImageColor.getcolor(color, "RGB")
    assert len(color_map) != 0, 'Invalid common type: {}'.format(common_type)
    
    return color_map


def get_color_map(common_type='default'):
    return common_color_map(common_type)


def colorize(mask, base_image=None, common_type='default'):
    color_map = get_color_map(common_type=common_type)

    # add turquoise sky color to color map since it is not present in nadir-facing LULC.
    color_map[255] = np.array([1, 204, 250], dtype=np.uint8)

    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)

    for i in np.unique(mask):
        index_mask = mask == i
        color_mask[index_mask] = color_map[i]

    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    # If base image is provided, overlay the color mask on top of it using alpha channel
    overlay_img = None
    if base_image is not None:
        overlay_img = cv2.addWeighted(base_image, 0.5, color_mask, 0.5, 0)

    return color_mask, overlay_img


common_classes_name2idx = {
    'water': 0,
    'trees': 1,
    'low_vegetation': 2,
    'built': 3,
    'ground': 4,
    'sky': 5,
    'dont_care': -1,
}

common_cartd_name2idx = {
    'bare_ground': 4,
    'rocky_terrain': 4,
    'developed_structures': 3,
    'road': 3,
    'shrubs': 2,
    'trees': 1,
    'sky': 5,
    'water': 0,
    'vehicles': -1,
    'person': -1,
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

more_common_cartd_name2idx = {
    'bare_ground': 3,
    'rocky_terrain': 3,
    'developed_structures': 2,
    'road': 2,
    'shrubs': 1,
    'trees': 1,
    'sky': 4,
    'water': 0,

    'vehicles': -1,
    'person': -1,
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

most_common_cartd_name2idx = {
    'bare_ground': 1,
    'rocky_terrain': 1,
    'developed_structures': 1,
    'road': 1,
    'shrubs': 1,
    'trees': 1,
    'sky': 2,
    'water': 0,

    'vehicles': -1,
    'person': -1,
}

cartd_idx2name = {
    0: 'bare_ground',
    1: 'rocky_terrain',
    2: 'developed_structures',
    3: 'road',
    4: 'shrubs',
    5: 'trees',
    6: 'sky',
    7: 'water',
    8: 'vehicles',
    9: 'person'
}


def get_converted_gt_mapping(old_idx2name, new_name2idx):
    new_class_idx_set = set()
    new_mapping = {}
    for k, v in old_idx2name.items():
        new_idx = new_name2idx[v]
        new_mapping[k] = new_idx

        new_class_idx_set.add(new_idx)
        print('({}) {} -> {}'.format(v, k, new_idx))

    for i in range(0, max(new_name2idx.values()) + 1):
        assert i in new_class_idx_set, 'Class index {} not found in new class set'.format(i)

    return new_mapping


def commonize_gt_labels(mask, class_mapping):
    new_mask = np.zeros_like(mask).astype(np.int32) - 1

    # The GT mask has 2 classes that have been set to -1 (dont care) so we should keep them as is
    for i in np.unique(mask):
        if i >= 0:
            new_mask[mask == i] = class_mapping[i]
    return new_mask
