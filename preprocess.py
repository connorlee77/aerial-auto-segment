import os
import tqdm
import argparse
import logging

import rasterio
import rasterio.plot
import rasterio.merge
import rasterio.mask
import shapely

# from reproject_raster import reproject_raster
from utils.rasterio_utils import mask_raster_by_shapely, reproject_raster_v2, save_raster_preview_as_png

# outputs/preprocessed/{epsg-xxxxx}/{place_name}/naip/{GSD}/mosaic.tiff
# outputs/preprocessed/{epsg-xxxxx}/{place_name}/dem/{GSD}/mosaic.tiff
# outputs/preprocessed/{epsg-xxxxx}/{place_name}/dsm/{GSD}/mosaic.tiff
# outputs/preprocessed/{epsg-xxxxx}/{place_name}/dynamicworld/{GSD}/mosaic.tiff


def reproject_resample_rasters(spatial_res, output_folder, place_name, data_path, force_reproject=False, save_raster_previews=False):
    """
        Reproject, resample, and mask naip/dsm/dem/dem_1m/dw rasters according to DynamicWorld label.
        ### Parameters:
            label_raster_path (str): Filepath to DynamicWorld label raster.
            spatial_res (float): Spatial resolution of output raster. If None, then use original resolution of each raster.
            output_folder (str): Path to output folder.
            place_name (str): Name of place (castaic lake, colorado river, etc...).
            data_path (str): Path to base data folder, i.e `/data/microsoft_planetary_computer`.
            force_reproject (bool): Whether to force reprojecting an existing output raster.
        ### Returns:
            None
    """

    # Name of directory to store reprojected rasters of different resolutions
    res_folder = '{:.1f}'.format(
        spatial_res) if spatial_res is not None else 'original_resolution'

    # Dictionary of dataset types and their interpolation methods
    dataset_type_dict = {
        'naip': dict(interpolation=rasterio.enums.Resampling.cubic),
        'planet': dict(interpolation=rasterio.enums.Resampling.cubic),
        'dem': dict(interpolation=rasterio.enums.Resampling.bilinear),
        'dem_1m': dict(interpolation=rasterio.enums.Resampling.bilinear),
        'dsm': dict(interpolation=rasterio.enums.Resampling.cubic),
        'dynamicworld': dict(interpolation=rasterio.enums.Resampling.nearest),
        'chesapeake_bay_lc': dict(interpolation=rasterio.enums.Resampling.nearest),
    }

    # Read label raster data (dynamic world labels)
    label_raster_path = os.path.join(
        data_path, 'dynamicworld', place_name, 'mosaic', 'mosaic.tiff')
    with rasterio.open(label_raster_path, 'r') as dw_labels:
        target_polygon = shapely.geometry.box(*dw_labels.bounds)
        target_crs = dw_labels.crs
        epsg_code = target_crs.to_epsg()

    for dataset_type, obj in dataset_type_dict.items():

        # Create filepaths for original mosaics (in their own CRS).
        original_path = os.path.join(
            data_path, dataset_type, place_name, 'mosaic', 'mosaic.tiff')
        if not os.path.exists(original_path):
            logging.warning(
                '{} does not exist, skipping...'.format(original_path))
            continue

        # Create filepath to the non-existing reprojected, resampled, and masked mosaic.
        reprojected_path = os.path.join(output_folder, 'epsg-{}'.format(
            epsg_code), place_name, dataset_type, res_folder, 'mosaic.tiff')
        if not os.path.exists(reprojected_path) or force_reproject:
            os.makedirs(os.path.dirname(reprojected_path), exist_ok=True)
            # Reproject, resample, and mask raster
            reproject_raster_v2(original_path, reprojected_path, dst_crs=target_crs,
                                spatial_res=spatial_res, interpolation=obj['interpolation'])
            mask_raster_by_shapely(
                reprojected_path, reprojected_path, target_polygon)
            if save_raster_previews:
                save_raster_preview_as_png(reprojected_path, chesapeake_bay=(dataset_type == 'chesapeake_bay_lc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial_res', type=float, default=None,
                        help='Spatial resolution of output raster. If None, then use original resolution of each raster.')
    parser.add_argument('--place_name', type=str, help='Name of place (castaic lake, colorado river, etc...)',
                        choices=['castaic_lake', 'colorado_river', 'big_bear_lake', 
                                 'duck', 'kentucky_river', 'clinton', 
                                 'virginia_beach_creeds', 'virginia_beach_false_cape_landing'])
    parser.add_argument('--data_path', type=str, default='/data/microsoft_planetary_computer/',
                        help='Path to base data folder, i.e /data/microsoft_planetary_computer')
    parser.add_argument('--force_reproject', action='store_true',
                        help='Whether to force reprojecting an existing output raster.')
    parser.add_argument('--save_raster_previews', action='store_true',
                        help='Whether to save raster previews as small pngs.')

    args = parser.parse_args()
    print(args)
    # Create output folders
    output_folder = os.path.join(args.data_path, 'outputs', 'preprocessed')
    os.makedirs(output_folder, exist_ok=True)

    reproject_resample_rasters(args.spatial_res,
                               output_folder, args.place_name, args.data_path, args.force_reproject, args.save_raster_previews)
