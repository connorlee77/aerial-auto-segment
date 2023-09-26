import os
import argparse

from utils.rasterio_utils import save_raster_preview_as_png, merge_rasters


def merge_all_rasters(data_path, dataset_type, save_preview=False):
    """
        Merge all rasters in `PLACE` subdirectories into respective single tif raster for each place. 
        Used to merge original (not reprojected) rasters only, i.e. after downloading tiles.
        ### Parameters:
            data_path (str): Path to dirctory of tif tiles to merge. Directory structure should be: /.../{PLACE}/tiles/*.tif
            dataset_type (str): Dataset type (naip, dem, dem_1m, dsm, dw).
            save_preview (bool): Whether to save preview of merged raster.
        ### Returns:
            None
    """
    full_data_path = os.path.join(data_path, dataset_type)
    for place in os.listdir(full_data_path):
        merge_rasters(os.path.join(full_data_path, place, 'tiles'),
                      os.path.join(full_data_path, place, 'mosaic'))
        if save_preview:
            save_raster_preview_as_png(os.path.join(
                full_data_path, place, 'mosaic', 'mosaic.tiff'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/microsoft_planetary_computer/',
                        help='Path to dirctory of tif tiles to merge. Directory structure should be: /DATA_PATH/.../tiles/*.tif')
    parser.add_argument('--dataset_type', type=str, choices=['naip', 'dem', 'dem_1m', 'dsm', 'dynamicworld'],
                        help='Dataset type (naip, dem, dem_1m, dsm, dynamicworld).')
    parser.add_argument('--save_preview', action='store_true',
                        help='Whether to save png preview of merged raster.')
    args = parser.parse_args()

    merge_all_rasters(args.data_path, args.dataset_type, args.save_preview)
