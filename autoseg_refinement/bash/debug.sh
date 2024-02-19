REPOS_DIR=/home/connor/repos
LULC_TYPE=dynamicworld
D3_MODEL=dsm
RESOLUTION=1.0
REFINEMENT_TYPE=none
PLACE=2022-12-20_Castaic_Lake
TRAJECTORY=flight4


unrefined_path=${LULC_TYPE}/${D3_MODEL}/${RESOLUTION}/${REFINEMENT_TYPE}/${PLACE}/${TRAJECTORY}
sam_predicted_mask_dir=${REPOS_DIR}/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks

python sam_refine.py \
--sam_predicted_mask_dir $sam_predicted_mask_dir \
--unrefined_semantic_mask_dir ${REPOS_DIR}/aerial-auto-segment/autosegment/outputs/${unrefined_path} \
--data_dir /data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/ \
--class_set $LULC_TYPE \
--commonize \
--output_dir outputs/${unrefined_path}
