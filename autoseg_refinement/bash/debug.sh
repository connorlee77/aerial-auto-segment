REPOS_DIR=/home/connor/repos
LULC_TYPE=dynamicworld
D3_MODEL=dem
RESOLUTION=1.0
REFINEMENT_TYPE=none
# PLACE=2022-12-20_Castaic_Lake
# TRAJECTORY=flight4
PLACE=2023-03-XX_Duck
TRAJECTORY=ONR_2023-03-22-14-41-46


unrefined_path=${LULC_TYPE}/${D3_MODEL}/${RESOLUTION}/${REFINEMENT_TYPE}/${PLACE}/${TRAJECTORY}
sam_predicted_mask_dir=${REPOS_DIR}/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p35_cartd_labeled_sam_png_masks

python sam_refine.py \
--sam_predicted_mask_dir $sam_predicted_mask_dir \
--unrefined_semantic_mask_dir ${REPOS_DIR}/aerial-auto-segment/autosegment/outputs/${unrefined_path} \
--data_dir /data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/ \
--class_set $LULC_TYPE \
--commonize \
--commonize_to default \
--output_dir test123/${unrefined_path}



# taskset --cpu-list 30-45 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/cartd_labeled_sam_png_masks outputs/open_sam_default

# taskset --cpu-list 15-29 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks outputs/open_sam_boxnms_0p50

# taskset --cpu-list 0-14 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p35_cartd_labeled_sam_png_masks outputs/open_sam_boxnms_0p35


# taskset --cpu-list 30-45 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/cartd_labeled_sam_png_masks outputs/more_common/open_sam_default more

# taskset --cpu-list 15-29 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks outputs/more_common/open_sam_boxnms_0p50 more

# taskset --cpu-list 0-14 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p35_cartd_labeled_sam_png_masks outputs/more_common/open_sam_boxnms_0p35 more


# taskset --cpu-list 50-65 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/cartd_labeled_sam_png_masks outputs/most_common/open_sam_default most

# taskset --cpu-list 66-80 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks outputs/most_common/open_sam_boxnms_0p50 most

# taskset --cpu-list 81-95 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p35_cartd_labeled_sam_png_masks outputs/most_common/open_sam_boxnms_0p35 most



# taskset --cpu-list 15-29 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks outputs/open_sam_boxnms_0p50 default

# taskset --cpu-list 30-44 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks outputs/more_common/open_sam_boxnms_0p50 more

# taskset --cpu-list 45-59 bash bash/refine_seg.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks outputs/most_common/open_sam_boxnms_0p50 most



taskset --cpu-list 15-29 bash bash/refine_seg_jitter.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks jitter_outputs/open_sam_boxnms_0p50 default

taskset --cpu-list 30-44 bash bash/refine_seg_jitter.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks jitter_outputs/more_common/open_sam_boxnms_0p50 more

taskset --cpu-list 45-59 bash bash/refine_seg_jitter.sh /home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks jitter_outputs/most_common/open_sam_boxnms_0p50 most
