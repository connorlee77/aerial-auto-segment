import argparse

import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_raster(input_filepath, output_filepath, dst_crs):
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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-raster-path', type=str, help='ex: dsm/castaiclake/mosaic/mosaic.tiff')  
    parser.add_argument('--output-raster-path', type=str, help='ex: dsm/castaiclake/mosaic/mosaic_epsg32611.tiff')
    parser.add_argument('--dst-crs', type=str, help='ex: EPSG:32611')
    args = parser.parse_args()
    print(args)

    reproject_raster(args.input_raster_path, args.output_raster_path, args.dst_crs)

    