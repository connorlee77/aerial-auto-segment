import argparse
from utils.rasterio_utils import reproject_raster, save_raster_preview_as_png

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-raster-path', type=str,
                        help='ex: dsm/castaiclake/mosaic/mosaic.tiff')
    parser.add_argument('--output-raster-path', type=str,
                        help='ex: dsm/castaiclake/mosaic/mosaic_epsg32611.tiff')
    parser.add_argument('--dst-crs', type=str, help='ex: EPSG:32611')
    args = parser.parse_args()
    print(args)
    # TODO: Something is wrong with reprojection of dynamic world rasters, likely due to nodata field.
    reproject_raster(args.input_raster_path,
                     args.output_raster_path, args.dst_crs)
    save_raster_preview_as_png(args.output_raster_path)
