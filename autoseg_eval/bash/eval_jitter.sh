refined_dir=/home/connor/repos/aerial-auto-segment/autoseg_refinement/jitter_outputs
gt_dir=/data/onr-thermal/cogito-annotation/converted-cogito-annotations-final
output_dir=./jitter_outputs

# common_type=more_common # more_common, most_common
# seg_src=open_sam_boxnms_0p35
# lulc_type=dynamicworld
# d3_type=dsm
# res=1.0
# refinement_type=crf_naip_naip-nir_surface_height

# dataset=2021-09-09-KentuckyRiver
# trajectory=flight1-1

COMMON_TYPE_LIST=(common more_common most_common)
SEG_SRC_LIST=(
    # open_sam_boxnms_0p35
    open_sam_boxnms_0p50
    # open_sam_default
    # sam_boxnms_0p35
    # sam_boxnms_0p50
    # sam_default
)
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
    # '10.0'
)

# gps_jitter_list=(0.5 1.0 2.0 3.0 5.0 10.0 25.0 50.0)
gps_jitter_list=(0.0)
# orientation_jitter_list=(0.0)
# altitude_jitter_list=(0.25 0.5 1.0 2.0 3.0 5.0 10.0 20.0)
altitude_jitter_list=(0.0)
orientation_jitter_list=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
# orientation_jitter_list=(0.0)
for ori in ${orientation_jitter_list[@]}; do
    for alt in ${altitude_jitter_list[@]}; do
        for gps in ${gps_jitter_list[@]}; do
            for res in ${RESOLUTION_LIST[@]}; do
                for refinement_type in ${REFINEMENT_TYPE_LIST[@]}; do
                    for d3_type in ${D3_TYPE_LIST[@]}; do
                        for lulc_type in ${LULC_TYPE_LIST[@]}; do
                            for common_type in ${COMMON_TYPE_LIST[@]}; do
                                for seg_src in ${SEG_SRC_LIST[@]}; do

                                    # Duck
                                    dataset=2023-03-XX_Duck
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
                                            python eval.py \
                                                --mask_dir ${refined_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory} \
                                                --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                                                --common_type ${common_type} \
                                                --output_dir ${output_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory}
                                        ) &
                                    done

                                    # Castaic Lake
                                    dataset=2022-12-20_Castaic_Lake
                                    TRAJECTORIES=('flight4')
                                    for trajectory in ${TRAJECTORIES[@]}; do
                                        (
                                            python eval.py \
                                                --mask_dir ${refined_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory} \
                                                --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                                                --common_type ${common_type} \
                                                --output_dir ${output_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory}
                                        ) &
                                    done

                                    # Colorado River
                                    dataset=2022-05-15_ColoradoRiver
                                    TRAJECTORIES=(
                                        flight2
                                        flight3
                                        flight4
                                    )
                                    for trajectory in ${TRAJECTORIES[@]}; do
                                        (
                                            python eval.py \
                                                --mask_dir ${refined_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory} \
                                                --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                                                --common_type ${common_type} \
                                                --output_dir ${output_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory}
                                        ) &
                                    done

                                    # Kentucky River
                                    dataset=2021-09-09-KentuckyRiver
                                    TRAJECTORIES=(
                                        flight1-1
                                        flight2-1
                                        flight3-1
                                    )
                                    for trajectory in ${TRAJECTORIES[@]}; do
                                        (
                                            python eval.py \
                                                --mask_dir ${refined_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory} \
                                                --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                                                --common_type ${common_type} \
                                                --output_dir ${output_dir}/${common_type}/${seg_src}/${lulc_type}/${gps}_${alt}_${ori}/${d3_type}/${res}/${refinement_type}/${dataset}/${trajectory}
                                        ) &
                                    done

                                done
                                wait
                            done

                        done

                    done
                done
            done
        done
    done
done
