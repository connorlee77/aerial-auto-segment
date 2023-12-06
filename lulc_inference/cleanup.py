import argparse
import os

import cv2
import numpy as np
import rasterio
import rasterio.plot
import skimage

from utils.utils import label2rgb


def load_data(data_path, epsg, location, lc_type, resolution, ir_src):
    '''
        Load the land cover data and the NIR source data

        Args:
            data_path (str): path to the data
            epsg (str): epsg code
            location (str): location name
            lc_type (str): land cover type
            resolution (float): resolution of the data
            ir_src (str): source of the NIR band

        Returns:
            lc_data (np.array): land cover data
            ir_data (np.array): NIR source data
            lc_profile (rasterio.profile): rasterio profile
    '''
    lc_path = os.path.join(data_path, '{}/{}/{}/{}/mosaic.tiff'.format(epsg,
                           location, lc_type, resolution))
    ir_path = os.path.join(data_path, '{}/{}/{}/{}/mosaic.tiff'.format(epsg,
                           location, ir_src, resolution))
    assert os.path.exists(lc_path), 'Land cover data does not exist: {}'.format(lc_path)
    assert os.path.exists(ir_path), 'NIR source data does not exist: {}'.format(ir_path)

    with rasterio.open(lc_path, 'r') as src:
        lc_data = src.read()
        lc_profile = src.profile

    with rasterio.open(ir_path, 'r') as src:
        ir_data = src.read()

    return lc_data, ir_data, lc_profile


def create_output_paths(data_path, epsg, location, lc_type, resolution):
    '''
        Create output directory and paths

        Args:
            data_path (str): path to the data
            epsg (str): epsg code
            location (str): location name
            lc_type (str): land cover type
            resolution (float): resolution of the data

        Returns:
            corrected_path (str): path to the corrected land cover raster
            corrected_preview_path (str): path to the corrected land cover RGB preview image
    '''
    corrected_lc_dir = os.path.join(data_path, '{}/{}/{}/{}'.format(epsg,
                                    location, '{}_corrected'.format(lc_type), resolution))
    os.makedirs(corrected_lc_dir, exist_ok=True)
    corrected_path = os.path.join(corrected_lc_dir, 'mosaic.tiff')
    corrected_preview_path = os.path.join(corrected_lc_dir, 'mosaic_preview.png')
    return corrected_path, corrected_preview_path


def create_water_mask_on_ir(water_threshold, ir_data, erosion_radius):
    '''
        Create a water mask based on the NIR band

        Args:
            water_threshold (float): threshold for water
            ir_data (np.array): NIR source data
            erosion_radius (int): erodes the water body contour by this many pixels. Pay attention to pixel resolution.

        Returns:
            eroded_mask (np.array): eroded water mask (H x W)
    '''

    ir = ir_data[-1]
    masked = ir < water_threshold
    eroded_mask = skimage.morphology.binary_erosion(masked, skimage.morphology.disk(erosion_radius))
    return eroded_mask


def apply_water_mask(lc_data, eroded_mask, water_index):
    '''
        Apply the water mask to the land cover data

        Args:
            lc_data (np.array): land cover data (N x H x W)
            eroded_mask (np.array): eroded water mask (H x W)
            water_index (int): index of the water class

        Returns:
            lc_data (np.array): corrected land cover data (N x H x W)
    '''
    _, H, W = lc_data.shape
    He, We = eroded_mask.shape
    assert abs(He - H) <= 1, 'Size difference to large: He: {}, H: {}'.format(He, H)
    assert abs(We - W) <= 1, 'Size difference to large: We: {}, W: {}'.format(We, W)

    # Resize the mask to the same size as the lc_data
    eroded_mask = cv2.resize(eroded_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # Invert the softmax back to logits
    raw_lc = np.log(lc_data[:-1, :, :] + 1e-8)

    # Set the water logits to anything above 0 to garuantee post-softmax maximum.
    raw_lc[water_index, eroded_mask == 1] = 0.5
    softmax_lc = np.exp(raw_lc) / np.sum(np.exp(raw_lc), axis=0)

    # Set new LULC data values
    lc_data[:-1, :, :] = softmax_lc
    lc_data[-1, :, :] = np.argmax(softmax_lc, axis=0)
    return lc_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/microsoft_planetary_computer/outputs/preprocessed/')
    parser.add_argument('--epsg', type=str, default='epsg-32618')
    parser.add_argument('--location', type=str, default='duck')
    parser.add_argument('--lc_type', type=str, default='chesapeake_bay_swin_crossentropy_lc_naip',
                        choices=['chesapeake_bay_swin_crossentropy_lc_naip', 'open_earth_map_unet_lc_naip'])
    parser.add_argument('--resolution', type=str, default='10.0', choices=['0.6', '1.0', '2.0', '3.0', '5.0', '10.0'])
    parser.add_argument('--water_threshold', type=float, default=None)
    parser.add_argument('--ir_src', type=str, default='naip',
                        choices=['naip', 'planet'], help='Which source to pull NIR band from')

    args = parser.parse_args()
    print(args)

    if 'chesapeake_bay_swin' in args.lc_type:
        water_idx = 0
        color_set = 'chesapeake-bay'
    elif 'open_earth_map' in args.lc_type:
        water_idx = 5
        color_set = 'open-earth-map'
    else:
        raise Exception('Unknown lc_type: {}'.format(args.lc_type))

    if args.ir_src == 'naip':
        erosion_radius = 2
    elif args.ir_src == 'planet':
        erosion_radius = 5

    # Read input data sources
    lc_data, ir_data, lc_profile = load_data(
        args.data_path, args.epsg, args.location, args.lc_type, args.resolution, args.ir_src)

    # Create output directory and paths
    corrected_path, corrected_preview_path = create_output_paths(
        args.data_path, args.epsg, args.location, args.lc_type, args.resolution)

    if args.water_threshold is not None:
        eroded_mask = create_water_mask_on_ir(args.water_threshold, ir_data, erosion_radius)
        lc_data = apply_water_mask(lc_data, eroded_mask, water_idx)
    else:
        print('Water threshold not provided. Saving without water mask')

    # Write the corrected raster
    with rasterio.open(corrected_path, 'w', **lc_profile) as dst:
        dst.write(lc_data)

    # Save the corrected RGB preview image
    lc_labels = lc_data[-1]
    lc_rgb = label2rgb(lc_labels, dataset=color_set)
    cv2.imwrite(corrected_preview_path, cv2.cvtColor(lc_rgb, cv2.COLOR_BGR2RGB))
