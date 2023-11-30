DATA_PATH=$1
PLACE_NAME=$2

# Original
python preprocess.py \
--place_name $PLACE_NAME \
--data_path $DATA_PATH \
--force_reproject \
--save_raster_previews

# NAIP only testing
# python preprocess.py \
# --spatial_res 0.6 \
# --place_name $PLACE_NAME \
# --data_path $DATA_PATH \
# --force_reproject \
# --save_raster_previews

# DEM 1m only testing
python preprocess.py \
--spatial_res 1.0 \
--place_name $PLACE_NAME \
--data_path $DATA_PATH \
--force_reproject \
--save_raster_previews

# DSM only testing
python preprocess.py \
--spatial_res 2.0 \
--place_name $PLACE_NAME \
--data_path $DATA_PATH \
--force_reproject \
--save_raster_previews

# # Original
# python preprocess.py \
# --place_name castaic_lake \
# --data_path /data/microsoft_planetary_computer/ \
# --force_reproject \
# --save_raster_previews

# # NAIP testing
# python preprocess.py \
# --spatial_res 0.6 \
# --place_name castaic_lake \
# --data_path /data/microsoft_planetary_computer/ \
# --force_reproject \
# --save_raster_previews

# # DEM 1m testing
# python preprocess.py \
# --spatial_res 1.0 \
# --place_name castaic_lake \
# --data_path /data/microsoft_planetary_computer/ \
# --force_reproject \
# --save_raster_previews

# # DSM testing
# python preprocess.py \
# --spatial_res 2.0 \
# --place_name castaic_lake \
# --data_path /data/microsoft_planetary_computer/ \
# --force_reproject \
# --save_raster_previews