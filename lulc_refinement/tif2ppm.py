import os
import glob
import argparse

import sys
sys.path.append('../')

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from utils.rasterio_utils import save_raster_preview_as_ppm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="path to file")
    parser.add_argument("--filetype", type=str, choices=['dynamicworld', 'chesapeake_bay_lc', 'naip'])
    args = parser.parse_args()
    print(args)
    input_raster_path = args.filepath
    if args.filetype == 'dynamicworld':
        save_raster_preview_as_ppm(input_raster_path, chesapeake_bay=False, common=True)
    elif args.filetype == 'chesapeake_bay_lc':
        save_raster_preview_as_ppm(input_raster_path, chesapeake_bay=True, common=True)
    elif args.filetype == 'naip':
        save_raster_preview_as_ppm(input_raster_path, chesapeake_bay=False, common=False)
    else:
        raise ValueError("Invalid filetype")
    