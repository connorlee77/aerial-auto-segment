import os
import tqdm
import argparse
import logging
import glob

import numpy as np

import rasterio
import rasterio.plot
import rasterio.merge
import rasterio.mask
import shapely

from utils.rasterio_utils import save_raster_preview_as_png

# Mapping for DynamicWorld and Chesapeake Bay Landcover labels to a common label set.
dw_cb_2_common = {

    'dynamicworld': {
        0: 0,  # water
        1: 1,  # trees
        2: 2,  # grass
        3: 4,  # flooded_vegetation
        4: 2,  # crops
        5: 3,  # shrub_and_scrub
        6: 5,  # built
        7: 6,  # bare
        8: -1,  # snow_and_ice (throw away)
    },
    'chesapeake_bay': {
        1: 0,  # 'Water',
        2: 4,  # 'Emergent Wetlands',
        3: 1,  # 'Tree Canopy',
        4: 3,  # 'Scrub\Shrub',
        5: 2,  # 'Low Vegetation',
        6: 6,  # 'Barren',
        7: 5,  # 'Impervious Structures',
        8: 5,  # 'Other Impervious',
        9: 5,  # 'Impervious Roads',
        10: 1,  # 'Tree Canopy over Impervious Structures',
        11: 1,  # 'Tree Canopy over Other Impervious',
        12: 1,  # 'Tree Canopy over Impervious Roads',
    }
}


def inverse_mapping(input_dict):
    inv_map = {}
    for k, v in input_dict.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    return inv_map


def convert_dw_common(input_raster_path, output_raster_path, save_raster_previews=False):
    """
        Converts a DynamicWorld raster to common LULC labels. Probabilities are kept.
        ### Parameters:
            input_raster_path (str): Path to raster file
            output_raster_path (str): Path to save converted raster file
            save_raster_previews (bool): Whether to save raster previews as small pngs.
        ### Returns:
            None
    """

    dw2common = dw_cb_2_common['dynamicworld']
    common2dw = inverse_mapping(dw2common)

    # new number of classes. common2dw is 1-indexed.
    num_new_classes = len(list(filter(lambda x: x >= 0, common2dw.keys())))

    # Read raster
    with rasterio.open(input_raster_path) as src:

        data = src.read()
        C, H, W = data.shape
        assert C == 10, f'Expected 10 channels, got {C} channels'

        # Create new array to hold class probabilities and label; NA values will be filled from data during label conversion loop.
        new_data = np.zeros((num_new_classes + 1, H, W), dtype=np.float32)
        assert new_data.shape == (8, H, W), f'Expected shape (8, {H}, {W}), got {new_data.shape}'

        # Convert labels; take max probability if new label aggregates multiple old labels.
        for new_idx, old_indices in common2dw.items():
            if new_idx >= 0:
                new_data[new_idx] = np.max(data[old_indices, :, :], axis=0)
        new_data[7] = np.argmax(new_data[:7, :, :], axis=0)

        # Copy profile and update band count
        out_profile = src.profile.copy()
        out_profile.update({
            'count': 8,
        })
        # Write new raster
        with rasterio.open(output_raster_path, "w", **out_profile) as dst:
            dst.write(new_data)
        if save_raster_previews:
            save_raster_preview_as_png(output_raster_path, chesapeake_bay=False, common=True)


def convert_cb_common(input_raster_path, output_raster_path, save_raster_previews=False):
    """
        Converts a label-only (single channel) Chesapeake Bay LULC raster to common LULC labels.
        ### Parameters:
            input_raster_path (str): Path to raster file
            output_raster_path (str): Path to save converted raster file
            save_raster_previews (bool): Whether to save raster previews as small pngs.
        ### Returns:
            None
    """

    cb2common = dw_cb_2_common['chesapeake_bay']
    common2cb = inverse_mapping(cb2common)

    # Read raster
    with rasterio.open(input_raster_path) as src:

        data = src.read()
        C, H, W = data.shape
        assert C == 1, f'Expected 1 channels, got {C} channels'

        # Create new array to hold class label; copy old data to use same NA values.
        new_data = np.copy(data)

        # Convert labels
        for new_idx, old_indices in common2cb.items():
            for old_idx in old_indices:
                new_data[data == old_idx] = new_idx

        # Write new raster
        out_profile = src.profile.copy()
        with rasterio.open(output_raster_path, "w", **out_profile) as dst:
            dst.write(new_data)
        if save_raster_previews:
            save_raster_preview_as_png(output_raster_path, chesapeake_bay=False, common=True)


def convert_folder(data_path, save_raster_previews=False):
    """
        Converts DynamicWorld and Chesaepake Landcover rasters to common LULC labels. Assumes that the input rasters
        are under the directory path `.../outputs/preprocessed/**/*mosaic.tiff`
        ### Parameters:
            data_path (str): Path to base data folder
            save_raster_previews (bool): Whether to save raster previews as small pngs.
        ### Returns:
            None
    """
    possible_label_pattern = os.path.join(data_path, 'outputs', 'preprocessed',
                                          '*', '*', 'REPLACE_ME', '*', 'mosaic.tiff')

    dw_label_mosaic_paths = glob.glob(possible_label_pattern.replace('REPLACE_ME', 'dynamicworld'))
    chesapeake_label_mosaic_paths = glob.glob(possible_label_pattern.replace('REPLACE_ME', 'chesapeake_bay_lc'))

    for input_raster_path in tqdm.tqdm(dw_label_mosaic_paths):
        output_raster_path = input_raster_path.replace('mosaic.tiff', 'converted_mosaic.tiff')
        convert_dw_common(input_raster_path, output_raster_path, save_raster_previews=save_raster_previews)

    for input_raster_path in tqdm.tqdm(chesapeake_label_mosaic_paths):
        output_raster_path = input_raster_path.replace('mosaic.tiff', 'converted_mosaic.tiff')
        convert_cb_common(input_raster_path, output_raster_path, save_raster_previews=save_raster_previews)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/microsoft_planetary_computer/',
                        help='Path to base data folder, i.e /data/microsoft_planetary_computer')
    parser.add_argument('--save_raster_previews', action='store_true',
                        help='Whether to save raster previews as small pngs.')

    args = parser.parse_args()
    print(args)

    convert_folder(args.data_path, args.save_raster_previews)
