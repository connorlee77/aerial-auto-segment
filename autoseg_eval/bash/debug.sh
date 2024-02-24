refined_dir=/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/
gt_dir=/data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/
output_dir=./outputs/

common_type=more_common # more_common, most_common
seg_src=open_sam_boxnms_0p35
lulc_type=dynamicworld
d3_type=dsm
res=1.0
refinement_type=crf_naip_naip-nir_surface_height

dataset=2021-09-09-KentuckyRiver
trajectory=flight1-1

python eval.py \
--mask_dir ${refined_dir}/${common_type}/${seg_src}/${lulc_type}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory} \
--gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
--common_type ${common_type} \
--output_dir ${output_dir}/${common_type}/${seg_src}/${lulc_type}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory} \