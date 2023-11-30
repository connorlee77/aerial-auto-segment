import glob
import logging
import os

import cv2
import numpy as np
import rasterio
import rasterio.mask
import rasterio.merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from utils.draw import (colorize_chesapeake_cvpr_landcover_label,
                        colorize_chesapeake_landcover_label,
                        colorize_common_landcover_label,
                        colorize_dynamic_world_label,
                        colorize_open_earth_map_landcover_label)


def does_raster_exist(filepath):
    if not os.path.exists(filepath):
        logging.warning('{} does not exist.'.format(filepath))
        return False
    else:
        return True


def ensure_single_crs(rasterio_tiles, tile_dir):
    """
        Ensure all rasters have the same CRS.
        ### Parameters:
            rasterio_tiles (list): List of rasterio datasets.
        ### Returns:
            None
    """
    crs = rasterio_tiles[0].crs
    for tile in rasterio_tiles:
        if tile.crs != crs:
            logging.warning('CRS mismatch: {} != {}. Reprojecting all rasters in {} to epsg:4326 (lat,lng)'.format(
                tile_dir, tile.crs, crs))
            reproject_rasters_batch(tile_dir, crs='EPSG:4326')
            break


def get_tif_files(tile_dir):
    """
        Get list of tif files in a directory.
        ### Parameters:
            tile_dir (str): Directory containing tifs.
        ### Returns:
            files (list): List of tif files.
    """
    files = []
    for ext in ['tif', 'tiff', 'TIF', 'TIFF']:
        files += glob.glob(os.path.join(tile_dir, '*.{}'.format(ext)))
    return files


def unmask_dynamic_world_raster(input_filepath, output_filepath, fill_value):
    """
        Fill nodata values in dynamic world raster with a specified value.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
            output_filepath (str): Filepath to output raster.
            fill_value (float): Value to fill nodata values with.
        ### Returns:
            None
    """
    if not does_raster_exist(input_filepath):
        return

    with rasterio.open(input_filepath) as src:
        data = src.read()
        np.nan_to_num(data, copy=False, nan=fill_value)
        kwargs = src.profile.copy()
        kwargs.update({
            'nodata': fill_value,
        })

        with rasterio.open(output_filepath, 'w', **kwargs) as dst:
            dst.write(data)


def unmask_dynamic_world_rasters_batch(tile_dir, fill_value=-9999.0):
    """
        Fill nodata values in dynamic world rasters in a directory with a specified value.
        See `unmask_dynamic_world_raster` for more details.
    """
    logging.warning('Unmasking (filling nodata values) DynamicWorld raster...')
    files = get_tif_files(tile_dir)
    for f in files:
        unmask_dynamic_world_raster(f, f, fill_value)


def merge_rasters(tile_dir, save_dir):
    """
        Merge tif rasters in a directory into a single tif raster.
        ### Parameters:
            tile_dir (str): Directory containing tifs to merge.
            save_dir (str): Directory to save merged tif.
        ### Returns:
            None
    """
    files = get_tif_files(tile_dir)

    if len(files) == 0:
        logging.warning('No files found in {}'.format(tile_dir))
    else:
        dss = [rasterio.open(f) for f in sorted(files)]
        # Make sure all rasters have the same CRS before merging. If not, reproject to epsg:4326 (lat,lng)
        ensure_single_crs(dss, tile_dir)
        ds, tform = rasterio.merge.merge(dss, nodata=0, method='first')

        out_meta = dss[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": ds.shape[1],
            "width": ds.shape[2],
            "transform": tform,
            "crs": dss[0].crs
        })

        os.makedirs(save_dir, exist_ok=True)
        mosaic_save_path = os.path.join(save_dir, 'mosaic.tiff')
        with rasterio.open(mosaic_save_path, 'w', **out_meta) as dest:
            dest.write(ds)


def reproject_raster(input_filepath, output_filepath, dst_crs):
    """
        Reproject raster to a different coordinate reference system (CRS). Does not perform any resampling.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
            output_filepath (str): Filepath to output raster.
            dst_crs (rasterio.crs.CRS): Target CRS.
        ### Returns:
            None
    """
    if not does_raster_exist(input_filepath):
        return

    with rasterio.open(input_filepath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_filepath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    num_threads=32,
                    warp_mem_limit=1024)


def reproject_rasters_batch(tile_dir, crs='EPSG:4326'):
    """
        Reproject rasters in a directory. See `reproject_raster` for more details.
    """
    files = get_tif_files(tile_dir)
    for f in files:
        reproject_raster(f, f, crs)


def mask_raster_by_shapely(input_filepath, output_filepath, polygon):
    """
        Crop raster by shapely polygon.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
            output_filepath (str): Filepath to output raster.
            polygon (shapely.geometry.Polygon): Polygon to crop raster.
        ### Returns:
            None
    """
    logging.info('Masking (cropping) raster...')
    if not does_raster_exist(input_filepath):
        return

    with rasterio.open(input_filepath) as src:
        out_image, out_transform = rasterio.mask.mask(
            src, [polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(output_filepath, "w", **out_meta) as dst:
            dst.write(out_image)


def reproject_raster_v2(input_filepath, output_filepath, dst_crs, spatial_res, interpolation):
    """
        Reproject and resample raster.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
            output_filepath (str): Filepath to output raster.
            dst_crs (rasterio.crs.CRS): Target CRS.
            spatial_res (float): Spatial resolution of output raster. If None, then use original resolution of each raster.
            interpolation (rasterio.enums.Resampling): Interpolation method.
        ### Returns:
            None
    """
    logging.info('Reprojecting and resampling raster...')
    if not does_raster_exist(input_filepath):
        return

    with rasterio.open(input_filepath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=spatial_res)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_filepath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=interpolation,
                    num_threads=32,
                    warp_mem_limit=1024)

            logging.info('Input:', input_filepath, src.shape,
                         src.crs, src.crs.linear_units, src.res)
            logging.info('Output:', output_filepath, dst.shape,
                         dst.crs, dst.crs.linear_units, dst.res)


def save_raster_preview_as_png(input_filepath, chesapeake_bay=False, common=False, chesapeake_cvpr=False, open_earth_map=False):
    """
        Save a preview of the raster as a png file.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
            chesapeake_bay (bool): Whether to colorize Chesapeake Bay Landcover labels.
            common (bool): Whether to colorize common LULC labels.
        ### Returns:
            None
    """
    logging.info('Saving raster preview as png...')
    if not does_raster_exist(input_filepath):
        return
    with rasterio.open(input_filepath, 'r') as x:
        img = x.read()
        C, _, _ = img.shape

        preview_filename = 'mosaic_preview.png'
        if common:
            preview_filename = 'converted_mosaic_preview.png'
            preview_img = img[-1]
            preview_img = colorize_common_landcover_label(preview_img)
        elif chesapeake_cvpr:
            preview_img = img[-1]
            preview_img = colorize_chesapeake_cvpr_landcover_label(preview_img)
        elif open_earth_map:
            preview_img = img[-1]
            preview_img = colorize_open_earth_map_landcover_label(preview_img)
        elif C == 4 or C == 3:
            # NAIP rasters
            preview_img = img.transpose(1, 2, 0)[:, :, :3]

        elif C == 1:
            if chesapeake_bay:
                preview_img = img[-1]
                preview_img = colorize_chesapeake_landcover_label(preview_img)
            else:
                # DSM/DEM rasters
                preview_img = np.clip(img[0], 0, None)
                preview_img = cv2.normalize(
                    preview_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif C == 10:
            # Dynamic World label raster
            preview_img = img[-1]
            preview_img = colorize_dynamic_world_label(preview_img)

        preview_img = cv2.resize(
            preview_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(os.path.dirname(input_filepath),
                    preview_filename), cv2.cvtColor(preview_img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_raster_preview_as_ppm(input_filepath, chesapeake_bay=False, common=False):
    """
        Save a preview of the raster as a ppm file.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
            chesapeake_bay (bool): Whether to colorize Chesapeake Bay Landcover labels.
            common (bool): Whether to colorize common LULC labels.
        ### Returns:
            None
    """
    logging.info('Saving raster preview as ppm...')
    if not does_raster_exist(input_filepath):
        return
    with rasterio.open(input_filepath, 'r') as x:
        img = x.read()
        C, _, _ = img.shape

        preview_filename = 'mosaic.ppm'
        if common:
            preview_filename = 'converted_mosaic.ppm'
            preview_img = img[-1]
            preview_img = colorize_common_landcover_label(preview_img)

        elif C == 4 or C == 3:
            # NAIP rasters
            preview_img = img.transpose(1, 2, 0)[:, :, :3]

        elif C == 1:
            if chesapeake_bay:
                preview_img = img[-1]
                preview_img = colorize_chesapeake_landcover_label(preview_img)
            else:
                # DSM/DEM rasters
                preview_img = np.clip(img[0], 0, None)
                preview_img = cv2.normalize(
                    preview_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        elif C == 10:
            # Dynamic World label raster
            preview_img = img[-1]
            preview_img = colorize_dynamic_world_label(preview_img)

        preview_img = cv2.resize(preview_img, (7781, 4911), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(os.path.dirname(input_filepath),
                    preview_filename), cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
