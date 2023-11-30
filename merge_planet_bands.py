import argparse
import os

import numpy as np
import rasterio
import tqdm
from utils.rasterio_utils import save_raster_preview_as_png


def stack_rgb_ir_raster(visual_path, four_band_path, out_path):
    """
        Stack visual and IR rasters into single RGBNIR raster.
        ### Parameters:
            visual_path (str): Path to visual raster.
            four_band_path (str): Path to IR raster.
            out_path (str): Path to output raster.
        ### Returns:
            None
    """
    with rasterio.open(visual_path, 'r') as visual_src:
        visual = visual_src.read([1, 2, 3])

        with rasterio.open(four_band_path, 'r') as four_band_src:
            four_band = four_band_src.read(4)
            assert four_band_src.crs == visual_src.crs, 'CRS mismatch'
            assert four_band_src.transform == visual_src.transform, 'Transform mismatch'
            rgb_nir = np.concatenate((visual, four_band[np.newaxis]), axis=0)
            with rasterio.open(out_path, 'w', **four_band_src.meta) as dst:
                dst.write(rgb_nir)


def stack_all_planet_rasters(data_path, dataset_type, save_preview=False):
    """
        Stack all planet rasters (after concatenating visual + IR) in `PLACE` subdirectories into respective single 
        tif raster for each place. Saves planet composite files as mosaics directly. 
        ### Parameters:
            data_path (str): Path to dirctory of tif tiles to merge. Directory structure should be: /.../{PLACE}/tiles/*.tif
            dataset_type (str): Dataset type (naip, dem, dem_1m, dsm, dw).
            save_preview (bool): Whether to save preview of merged raster.
        ### Returns:
            None
    """
    full_data_path = os.path.join(data_path, dataset_type)
    for place in tqdm.tqdm(os.listdir(full_data_path)):
        visual_path = os.path.join(full_data_path, place, 'visual', 'composite.tif')
        four_band_path = os.path.join(full_data_path, place, '4band', 'composite.tif')

        mosaic_dir = os.path.join(full_data_path, place, 'mosaic')
        os.makedirs(mosaic_dir, exist_ok=True)

        mosaic_path = os.path.join(mosaic_dir, 'mosaic.tiff')

        stack_rgb_ir_raster(visual_path, four_band_path, out_path=mosaic_path)
        if save_preview:
            save_raster_preview_as_png(os.path.join(mosaic_dir, 'mosaic.tiff'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/microsoft_planetary_computer/',
                        help='Path to dirctory of tif tiles to merge. Directory structure should be: /DATA_PATH/.../tiles/*.tif')
    parser.add_argument('--dataset_type', type=str, choices=['naip', 'dem', 'dem_1m', 'dsm', 'dynamicworld', 'planet'],
                        help='Dataset type (naip, dem, dem_1m, dsm, dynamicworld).')
    parser.add_argument('--save_preview', action='store_true',
                        help='Whether to save png preview of merged raster.')
    args = parser.parse_args()
    print(args)
    stack_all_planet_rasters(args.data_path, args.dataset_type, args.save_preview)
