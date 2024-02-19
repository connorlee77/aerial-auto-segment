dataset=$1
trajectory=$2
place=$3
d3_dtype=$4
epsg=$5
lulc_type=$6
resolution=$7
refinement_type=$8

python autoseg_v2.py \
    --data_path /data/onr-thermal/${dataset}/${trajectory} \
    --place ${place} \
    --d3_type ${d3_dtype} \
    --output_dir outputs/${lulc_type}/${d3_dtype}/${resolution}/${refinement_type}/${dataset}/${trajectory} \
    --epsg ${epsg} \
    --lulc_type ${lulc_type} \
    --resolution ${resolution} \
    --refinement_type ${refinement_type}
