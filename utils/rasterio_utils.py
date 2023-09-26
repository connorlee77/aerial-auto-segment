import os
import glob
import cv2
import numpy as np

import rasterio
import rasterio.mask
import rasterio.merge
from rasterio.warp import calculate_default_transform, reproject, Resampling


def does_raster_exist(filepath):
    if not os.path.exists(filepath):
        print('{} does not exist.'.format(filepath))
        return False
    else:
        return True


def merge_rasters(tile_dir, save_dir):
    """
        Merge tif rasters in a directory into a single tif raster.
        ### Parameters:
            tile_dir (str): Directory containing tifs to merge.
            save_dir (str): Directory to save merged tif.
        ### Returns:
            None
    """
    files = glob.glob(os.path.join(tile_dir, '*.tif'))
    if len(files) == 0:
        print('No files found in {}'.format(tile_dir))
    else:
        dss = [rasterio.open(f) for f in sorted(files)]
        ds, tform = rasterio.merge.merge(dss, nodata=0)

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
        Reproject raster.
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
                    resampling=Resampling.nearest)


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
    print('Masking (cropping) raster...')
    if not does_raster_exist(input_filepath):
        return

    print('Applying crop mask to raster...')
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
            print('Input:', input_filepath, src.shape,
                  src.crs, src.crs.linear_units, src.res)
            print('Output:', output_filepath, dst.shape,
                  dst.crs, dst.crs.linear_units, dst.res)


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
    print('Reprojecting and resampling raster...')
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

            print('Input:', input_filepath, src.shape,
                  src.crs, src.crs.linear_units, src.res)
            print('Output:', output_filepath, dst.shape,
                  dst.crs, dst.crs.linear_units, dst.res)


def save_raster_preview_as_png(input_filepath):
    """
        Save a preview of the raster as a png file.
        ### Parameters:
            input_filepath (str): Filepath to input raster.
        ### Returns:
            None
    """
    print('Saving raster preview as png...')
    if not does_raster_exist(input_filepath):
        return
    with rasterio.open(input_filepath, 'r') as x:
        img = x.read()
        C, _, _ = img.shape
        if C == 4 or C == 3:
            # NAIP rasters
            preview_img = img.transpose(1, 2, 0)[:, :, :3]
            preview_img = cv2.resize(
                preview_img, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
        elif C == 1:
            # DSM/DEM rasters
            preview_img = np.clip(img[0], 0, None)
            preview_img = cv2.normalize(
                preview_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            preview_img = cv2.resize(
                preview_img, (0, 0), fx=0.02, fy=0.02, interpolation=cv2.INTER_NEAREST)
        elif C > 4:
            # Dynamic World label raster
            preview_img = img[-1]
            preview_img = cv2.normalize(
                preview_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            preview_img = cv2.resize(
                preview_img, (0, 0), fx=0.02, fy=0.02, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(os.path.dirname(input_filepath),
                    'mosaic_preview.png'), cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
