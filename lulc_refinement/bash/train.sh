# n feature channels should be all of the channels combined, including surface, even though it is used separately
python train.py \
--base_dir /data/chesapeake_bay_lulc/outputs/preprocessed \
--epsg epsg-32618 \
--dataset clinton virginia_beach_creeds virginia_beach_false_cape_landing \
--resolution 1.0 \
--parallel_jobs 25 \
--n_trials 100 \
--unary_src dynamicworld \
--feature_set naip naip-nir dsm \
--nonconstant_kernel_parameters \
--n_feature_channels 5 \
--study_name chesapeake-bay-crf-tuning-boundary-loss-nir-dsm \
--boundary_loss \
--device_id 0 \
--visualize
