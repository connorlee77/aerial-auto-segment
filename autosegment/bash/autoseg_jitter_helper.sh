dataset=$1
trajectory=$2
place=$3
d3_dtype=$4
epsg=$5
lulc_type=$6
resolution=$7
refinement_type=$8
gps_jitter=$9
orientation_jitter=${10}
alt_jitter=${11}

python autoseg_v2.py \
    --data_path /data/onr-thermal/${dataset}/${trajectory} \
    --place ${place} \
    --d3_type ${d3_dtype} \
    --output_dir jitter_outputs/${lulc_type}/${gps_jitter}_${alt_jitter}_${orientation_jitter}/${d3_dtype}/${resolution}/${refinement_type}/${dataset}/${trajectory} \
    --epsg ${epsg} \
    --lulc_type ${lulc_type} \
    --resolution ${resolution} \
    --refinement_type ${refinement_type} \
    --jitter-gps-std ${gps_jitter} \
    --jitter-orientation-std ${orientation_jitter} \
    --jitter-alt-std ${alt_jitter}
