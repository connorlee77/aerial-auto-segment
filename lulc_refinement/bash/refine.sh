BASE_DIR=/data/microsoft_planetary_computer/outputs/preprocessed
EPSG=epsg-32611 # epsg-32616  epsg-32618
DATASET=big_bear_lake #  castaic_lake  colorado_river
RESOLUTION=1.0
UNARY_SRC=dynamicworld
FEATURE_SET='naip naip-nir naip-ndvi surface_height'

python refine.py \
--base_dir $BASE_DIR \
--epsg $EPSG \
--dataset $DATASET \
--resolution $RESOLUTION \
--unary_src $UNARY_SRC \
--feature_set $FEATURE_SET \
--theta_alpha $ALPHA \
--theta_alpha_z $ALPHA_Z \
--theta_betas $BETA \
--theta_gamma $GAMMA \
--theta_gamma_z $GAMMA_Z \
--w1 $W1 \
--w2 $W2 \
--output_dir lulc_refine_outputs/${EXP_NAME}/${EPSG}/${DATASET}/refined_lulc/${RESOLUTION}/${UNARY_SRC}/${FEATURE_SET// /_}
