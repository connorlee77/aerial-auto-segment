# segmentation_masks_path=/home/connor/repos/caltech-aerial-thermal-dataset/foundation_models/segmentation/boxnms0p5_cartd_labeled_sam_png_masks
segmentation_masks_path=$1 
output_dir=$2
commonize_to=$3

# taskset --cpu-list 0-14 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/felzenszwalb ./felzenszwalb_outputs default

# taskset --cpu-list 15-29 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/felzenszwalb ./felzenszwalb_outputs more

# taskset --cpu-list 30-44 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/felzenszwalb ./felzenszwalb_outputs most

# taskset --cpu-list 45-59 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/slic ./slic_outputs default

# taskset --cpu-list 60-74 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/slic ./slic_outputs more

# taskset --cpu-list 75-89 bash bash/refine_classical.sh /home/connor/repos/aerial-auto-segment/autoseg_classical_segmentation/slic ./slic_outputs most



# This works for any segmentations, even not SAM, as long as the unrefined_semantic_mask_dir is in the same format as the SAM predictions
unrefined_segmentations_dir=/home/connor/repos/aerial-auto-segment/autosegment/outputs_v2

D3_TYPE_LIST=('dem')
LULC_TYPE_LIST=(
    "dynamicworld"
    # "chesapeake_bay_swin_crossentropy_lc_naip_corrected"
    # "chesapeake_bay_swin_crossentropy_lc_planet"
    # "open_earth_map_unet_lc_naip_corrected"
    # "open_earth_map_unet_lc_planet"
)
REFINEMENT_TYPE_LIST=(
    'none'
    # 'crf_naip_naip-nir'
    # 'crf_naip_naip-nir_surface_height'
    # 'crf_planet'
    # 'crf_planet_surface_height'
)
RESOLUTION_LIST=(
    '1.0'
)

for resolution in ${RESOLUTION_LIST[@]}; do
    for refinement_type in ${REFINEMENT_TYPE_LIST[@]}; do
        for d3_dtype in ${D3_TYPE_LIST[@]}; do
            for lulc_type in ${LULC_TYPE_LIST[@]}; do

                case $lulc_type in
                *"dynamicworld"*)
                    class_set=dynamicworld
                    ;;
                "chesapeake_bay"*)
                    class_set=chesapeake
                    ;;
                "open_earth_map"*)
                    class_set=open_earth_map
                    ;;
                esac

                # Duck
                PLACE=2023-03-XX_Duck
                TRAJECTORIES=(
                    # ONR_2023-03-21-18-20-21 # something is wrong with this trajectory
                    ONR_2023-03-22-14-41-46
                    # ONR_2023-03-22-08-44-31 # no aligned.csv
                    ONR_2023-03-21-19-55-11
                    ONR_2023-03-21-14-06-04
                    ONR_2023-03-21-09-59-39
                )
                for trajectory in ${TRAJECTORIES[@]}; do
                    (
                    unrefined_path=${lulc_type}/${d3_dtype}/${resolution}/${refinement_type}/${PLACE}/${trajectory}
                    sam_predicted_mask_dir=$segmentation_masks_path
                    bash bash/refine_helper.sh $sam_predicted_mask_dir $unrefined_segmentations_dir $unrefined_path $class_set $output_dir $commonize_to 
                    ) &
                done

                # Castaic Lake
                PLACE=2022-12-20_Castaic_Lake
                TRAJECTORIES=('flight4')
                for trajectory in ${TRAJECTORIES[@]}; do
                    unrefined_path=${lulc_type}/${d3_dtype}/${resolution}/${refinement_type}/${PLACE}/${trajectory}
                    sam_predicted_mask_dir=$segmentation_masks_path
                    bash bash/refine_helper.sh $sam_predicted_mask_dir $unrefined_segmentations_dir $unrefined_path $class_set $output_dir $commonize_to
                done

                # Colorado River
                PLACE=2022-05-15_ColoradoRiver
                TRAJECTORIES=(
                    flight2
                    flight3
                    flight4
                )
                for trajectory in ${TRAJECTORIES[@]}; do
                (
                    unrefined_path=${lulc_type}/${d3_dtype}/${resolution}/${refinement_type}/${PLACE}/${trajectory}
                    sam_predicted_mask_dir=$segmentation_masks_path
                    bash bash/refine_helper.sh $sam_predicted_mask_dir $unrefined_segmentations_dir $unrefined_path $class_set $output_dir $commonize_to
                ) &
                done

                # Kentucky River
                PLACE=2021-09-09-KentuckyRiver
                TRAJECTORIES=(
                    flight1-1
                    flight2-1
                    flight3-1
                )
                for trajectory in ${TRAJECTORIES[@]}; do
                (
                    unrefined_path=${lulc_type}/${d3_dtype}/${resolution}/${refinement_type}/${PLACE}/${trajectory}
                    sam_predicted_mask_dir=$segmentation_masks_path
                    bash bash/refine_helper.sh $sam_predicted_mask_dir $unrefined_segmentations_dir $unrefined_path $class_set $output_dir $commonize_to
                ) &
                done
                wait

            done
        done
    done
done
