sam_predicted_mask_dir=$1
unrefined_segmentations_dir=$2
unrefined_path=$3
class_set=$4
output_dir=$5
commonize_to=$6

python sam_refine.py \
    --sam_predicted_mask_dir $sam_predicted_mask_dir \
    --unrefined_semantic_mask_dir ${unrefined_segmentations_dir}/${unrefined_path} \
    --data_dir /data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/ \
    --class_set $class_set \
    --commonize \
    --commonize_to $commonize_to \
    --output_dir $output_dir/${unrefined_path}
