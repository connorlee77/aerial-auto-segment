# /data/microsoft_planetary_computer/
# /data/cheseapeake_bay_lulc/
DATA_PATH=$1

python merge_tifs.py \
--data_path $DATA_PATH \
--dataset_type naip \
--save_preview

python merge_tifs.py \
--data_path $DATA_PATH \
--dataset_type dem \
--save_preview

python merge_tifs.py \
--data_path $DATA_PATH \
--dataset_type dem_1m \
--save_preview

python merge_tifs.py \
--data_path $DATA_PATH \
--dataset_type dsm \
--save_preview

python merge_tifs.py \
--data_path $DATA_PATH \
--dataset_type dynamicworld \
--save_preview
