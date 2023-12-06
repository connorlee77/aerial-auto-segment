# castaic lake - naip, 57.47599085746225
# big_bear_lake - planet, 1050.5703125
# colorado river - No correction
# kentucky river - planet, 1324.0589079436945
# duck = planet, 411.9718089421824
LC_TYPE=$1

for RESOLUTION in 0.6 1.0 2.0 3.0 5.0 10.0
do
    python cleanup.py \
    --data_path /data/microsoft_planetary_computer/outputs/preprocessed/ \
    --epsg epsg-32611 \
    --location colorado_river \
    --lc_type $LC_TYPE \
    --resolution $RESOLUTION

    python cleanup.py \
    --data_path /data/microsoft_planetary_computer/outputs/preprocessed/ \
    --epsg epsg-32611 \
    --location castaic_lake \
    --lc_type $LC_TYPE \
    --resolution $RESOLUTION \
    --water_threshold 57.47599085746225 \
    --ir_src naip

    python cleanup.py \
    --data_path /data/microsoft_planetary_computer/outputs/preprocessed/ \
    --epsg epsg-32611 \
    --location big_bear_lake \
    --lc_type $LC_TYPE \
    --resolution $RESOLUTION \
    --water_threshold 1050.5703125 \
    --ir_src planet

    python cleanup.py \
    --data_path /data/microsoft_planetary_computer/outputs/preprocessed/ \
    --epsg epsg-32618 \
    --location duck \
    --lc_type $LC_TYPE \
    --resolution $RESOLUTION \
    --water_threshold 411.97 \
    --ir_src planet

    python cleanup.py \
    --data_path /data/microsoft_planetary_computer/outputs/preprocessed/ \
    --epsg epsg-32616 \
    --location kentucky_river \
    --lc_type $LC_TYPE \
    --resolution $RESOLUTION \
    --water_threshold 1324.0589079436945 \
    --ir_src planet
done