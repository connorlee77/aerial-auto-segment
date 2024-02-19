from PIL import ImageColor
import numpy as np
import cv2

def dynamic_world_color_map():
    HEX_COLORS = [
        '#419BDF',  # water 0
        '#397D49',  # trees 1
        '#88B053',  # grass 2
        '#7A87C6',  # Flooded vegetation 3
        '#E49635',  # Crops 4 
        '#DFC35A',  # Shrub and scrub 5
        '#C4281B',  # built 6
        '#ffffff',  # bare 7
        '#B39FE1',  # snow and ice 8
    ]

    rgb_colors = [ImageColor.getcolor(c, "RGB") for c in HEX_COLORS]
    color_map = dict(zip(list(range(0, 9)), rgb_colors))
    return color_map


def chesapeake_cvpr_landcover_color_map():

    color_map = {
        # -1: np.array([0,  0, 0], dtype=np.uint8), # background
        0: np.array([0, 197, 255], dtype=np.uint8),  # water
        1: np.array([38, 115, 0], dtype=np.uint8),  # tree canopy and shrubs
        2: np.array([163, 255, 115], dtype=np.uint8),  # low vegetation
        3: np.array([255, 170, 0], dtype=np.uint8),  # barren
        4: np.array([156, 156, 156], dtype=np.uint8),  # impervious surfaces
        5: np.array([0, 0, 0], dtype=np.uint8),  # impervious roads
        6: np.array([178, 178, 178], dtype=np.uint8),  # aberdeen proving ground
    }

    return color_map


def open_earth_map_landcover_color_map():
    color_map = {
        0: np.array([128, 0, 0]),  # Bareland
        1: np.array([0, 255, 36]),  # Grass
        2: np.array([148, 148, 148]),  # Pavement
        3: np.array([255, 255, 255]),  # Road
        4: np.array([34, 97, 38]),  # Tree
        5: np.array([0, 69, 255]),  # Water
        6: np.array([75, 181, 73]),  # Cropland
        7: np.array([222, 31, 7]),  # buildings
    }

    return color_map

common_classes_name2idx = {
    'water': 0,
    'trees': 1,
    'low_vegetation': 2,
    'built': 3,
    'ground': 4,
    'sky': 5,
    'dont_care': -1,
}


def common_color_map():
    common_colors = {
        0: '#419BDF',  # water
        1: '#397D49',  # trees (tree canopy)
        2: '#88B053',  # low vegetation (shrubs, grass, crops, flooded vegetation)
        3: '#C4281B',  # built (impervious structures, other impervious, impervious roads)
        4: '#ffffff',  # ground (bare, barren)
        5: '#01CCFA',  # sky
        -1: '#B39FE1',  # dont care (snow and ice, aberdeen proving ground)
    }

    color_map = {}
    for idx, color in common_colors.items():
        color_map[idx] = ImageColor.getcolor(color, "RGB")
    return color_map

def get_color_map(set_name):
    if set_name == 'chesapeake':
        return chesapeake_cvpr_landcover_color_map()
    elif set_name == 'open_earth_map':
        return open_earth_map_landcover_color_map()
    elif set_name == 'dynamicworld':
        return dynamic_world_color_map()
    elif set_name == 'common':
        return common_color_map()
    else:
        raise ValueError(f'No color map found for {set_name}')
    
def colorize(mask, set_name, base_image=None):
    color_map = get_color_map(set_name)

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