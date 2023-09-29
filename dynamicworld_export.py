import os
import glob
import time
import argparse
import numpy as np
import cv2
import ee

from utils.utils import read_geojson, read_url_as_img

from utils.draw import colorize_dynamic_world_label


def save_sample_mosaic(annotation_collection, place_name, save_dir):
    print('Saving sample mosaic...')
    annotation_mosaic = annotation_collection.mosaic().clip(aoi)
    print("Bands: ", list(annotation_mosaic.bandNames().getInfo()))

    # Save a small image to visualize
    url = annotation_mosaic.select(['label']).getDownloadURL(
        {'scale': 10, 'min': 0, 'max': 9, 'format': 'NPY'})
    example_img = read_url_as_img(url)
    colorized_mask = colorize_dynamic_world_label(example_img['label'])

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, '{}_label_mosaic.png'.format(
        place_name)), cv2.cvtColor(colorized_mask, cv2.COLOR_RGB2BGR))


def export_collection(annotation_collection, place_name, dst_crs, aoi, dryrun=False):
    num_images = annotation_collection.size().getInfo()
    annotation_list = annotation_collection.toList(num_images)

    tasks = []
    for i in range(num_images):
        img = ee.Image(annotation_list.get(i)).float()
        date_str = str(ee.Date(img.get("system:time_start")
                               ).format('YYYY-MM-dd').getInfo())

        idx_str = str(i).zfill(5)
        cur_fname = place_name + '_' + idx_str + '_' + date_str
        print('Exporting ', cur_fname)

        if not dryrun:
            # task = ee.batch.Export.image.toDrive(
            #     image=img,
            #     region=aoi,
            #     crs=dst_crs,
            #     description=cur_fname,
            #     folder='dynamicworld-{}'.format(place_name),
            #     maxPixels=3e10,
            # )

            task = ee.batch.Export.image.toCloudStorage(
                image=img,
                description=cur_fname,
                bucket='ee-flying-goose',
                fileNamePrefix='dynamicworld/{}/{}'.format(
                    place_name, idx_str + '_' + date_str),
                region=aoi,
                crs=dst_crs,
                scale=10,  # meters / pixel
                maxPixels=3e10,
            )

            tasks.append(task)
            task.start()

            if i == 0:
                time.sleep(15)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson-path', type=str,
                        help='ex: dsm/castaiclake/mosaic/mosaic.tiff')
    parser.add_argument('--dst-crs', type=str, help='ex: EPSG:32611')
    parser.add_argument('--start-date', type=str,
                        help='YYYY-MM-DD, ex. castaic lake 2022-12-18')
    parser.add_argument('--end-date', type=str,
                        help='YYYY-MM-DD, ex. castaic lake 2022-12-23')
    parser.add_argument('--dryrun', action='store_true',
                        help='Dry run, do not export')
    parser.add_argument('--sample-mosaic-save-dir', default='.',
                        help='Directory to save sample mosaic')
    args = parser.parse_args()
    print(args)

    ee.Initialize()

    dst_crs = args.dst_crs
    startDate = args.start_date
    endDate = args.end_date

    place_name = os.path.basename(args.geojson_path).replace('.geojson', '')

    geojson = read_geojson(args.geojson_path)
    coords = geojson['geometry']['coordinates']
    aoi = ee.Geometry.Polygon(coords)

    annotation_collection = ee.ImageCollection(
        "GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi).filterDate(ee.Date(startDate), ee.Date(endDate))

    save_sample_mosaic(annotation_collection, place_name,
                       args.sample_mosaic_save_dir)
    export_collection(annotation_collection, place_name,
                      dst_crs, aoi, dryrun=args.dryrun)
