python train.py \
--base_dir /data/chesapeake_bay_lulc/outputs/preprocessed \
--epsg epsg-32618 \
--dataset clinton virginia_beach_creeds virginia_beach_false_cape_landing \
--resolution 1.0 \
--unary_src dynamicworld \
--feature_set naip \
--parallel_jobs 20 \
--cores-to-use 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
--n_trials 100 \
--study_name chesapeake-bay-crf-tuning-boundary-loss \
--boundary_loss \
--augment_boundary_loss \
--visualize
