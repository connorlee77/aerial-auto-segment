vfm_dir=/home/connor/repos/aerial-auto-segment/autoseg_vfm/aerial_autoseg_outputs
gt_dir=/data/onr-thermal/cogito-annotation/converted-cogito-annotations-final
output_dir=./vfm_outputs

COMMON_TYPE_LIST=(common more_common most_common)
SEG_SRC_LIST=(
    odise
)

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
                    --mask_dir ${vfm_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}/mask \
                    --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                    --common_type ${common_type} \
                    --output_dir ${output_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}
            ) &
        done

        # Castaic Lake
        dataset=2022-12-20_Castaic_Lake
        TRAJECTORIES=('flight4')
        for trajectory in ${TRAJECTORIES[@]}; do
            (
                python eval.py \
                    --mask_dir ${vfm_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}/mask \
                    --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                    --common_type ${common_type} \
                    --output_dir ${output_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}
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
                    --mask_dir ${vfm_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}/mask \
                    --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                    --common_type ${common_type} \
                    --output_dir ${output_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}
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
                    --mask_dir ${vfm_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}/mask \
                    --gt_dir ${gt_dir}/${dataset}/${trajectory}/masks/ \
                    --common_type ${common_type} \
                    --output_dir ${output_dir}/${common_type}/${seg_src}/${dataset}/${trajectory}
            ) &
        done
        wait
    done
done
