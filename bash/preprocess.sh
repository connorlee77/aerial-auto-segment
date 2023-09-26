# Original
python preprocess.py \
--place_name castaic_lake \
--label_raster_path /data/microsoft_planetary_computer/label_mosaic_v2.tiff \
--data_path /data/microsoft_planetary_computer/ \
--force_reproject \
--save_raster_previews

# NAIP testing
python preprocess.py \
--spatial_res 0.6 \
--place_name castaic_lake \
--label_raster_path /data/microsoft_planetary_computer/label_mosaic_v2.tiff \
--data_path /data/microsoft_planetary_computer/ \
--force_reproject \
--save_raster_previews

# DEM 1m testing
python preprocess.py \
--spatial_res 1.0 \
--place_name castaic_lake \
--label_raster_path /data/microsoft_planetary_computer/label_mosaic_v2.tiff \
--data_path /data/microsoft_planetary_computer/ \
--force_reproject \
--save_raster_previews

# DSM testing
python preprocess.py \
--spatial_res 2.0 \
--place_name castaic_lake \
--label_raster_path /data/microsoft_planetary_computer/label_mosaic_v2.tiff \
--data_path /data/microsoft_planetary_computer/ \
--force_reproject \
--save_raster_previews