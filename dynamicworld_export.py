import os
import glob
import time
import argparse
import numpy as np
import cv2
import ee

from utils.utils import read_geojson, read_url_as_img



def save_sample_mosaic(annotation_collection):
    annotation_mosaic = annotation_collection.mosaic().clip(aoi)
    print("Bands: ", list(annotation_mosaic.bandNames().getInfo()))

    # Save a small image to visualize
    url = annotation_mosaic.select(['label']).getDownloadURL({'scale' : 30, 'min':0, 'max':9, 'format' : 'NPY'})
    example_img = read_url_as_img(url)
    cv2.imwrite('example_label.png', np.uint8(example_img['label'] / 9.0 * 255))

def export_collection(annotation_collection, place_name, dst_crs, aoi):
    num_images = annotation_collection.size().getInfo()
    annotation_list = annotation_collection.toList(num_images)

    tasks = []
    for i in range(num_images):
        img = ee.Image(annotation_list.get(i)).float()
        date_str = str(ee.Date(img.get("system:time_start")).format('YYYY-MM-dd').getInfo())

        idx_str = str(i).zfill(5)
        cur_fname = place_name + '_' + idx_str + '_' + date_str
        print('Exporting ', cur_fname)

        task = ee.batch.Export.image.toDrive(
            image=img,
            region=aoi,
            crs=dst_crs,
            description=cur_fname,
            folder='dynamicworld-{}'.format(place_name),
            maxPixels=3e10,
        )
        
        tasks.append(task)
        task.start()

        if i == 0:
            time.sleep(15) 
        break
    
    while True:
        for t in tasks:
            print(t.status())
        time.sleep(5) 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson-path', type=str, help='ex: dsm/castaiclake/mosaic/mosaic.tiff')  
    parser.add_argument('--dst-crs', type=str, help='ex: EPSG:32611')
    parser.add_argument('--start-date', type=str, help='YYYY-MM-DD, ex. castaic lake 2022-12-18')
    parser.add_argument('--end-date', type=str, help='YYYY-MM-DD, ex. castaic lake 2022-12-23')
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

    annotation_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi).filterDate(ee.Date(startDate), ee.Date(endDate))
    save_sample_mosaic(annotation_collection)
    export_collection(annotation_collection, place_name, dst_crs, aoi)
