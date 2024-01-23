EXP_NAME=$1
ALPHA=$2
BETA=$3
GAMMA=$4
W1=$5
W2=$6
FEATURE_SET=$7
ALPHA_Z=$8
GAMMA_Z=$9

echo Running experiment: $EXP_NAME

echo Parameters:
echo theta_alpha: $ALPHA
echo theta_beta: $BETA
echo theta_gamma: $GAMMA
echo w1: $W1
echo w2: $W2
echo theta_alpha_z: $ALPHA_Z
echo theta_gamma_z: $GAMMA_Z
echo

BASE_DIR=/data/microsoft_planetary_computer/outputs/preprocessed/
RESOLUTIONs=('0.6' '1.0' '2.0' '3.0' '5.0')
UNARY_SRCs=('dynamicworld' 'chesapeake_bay_swin_crossentropy_lc_naip_corrected' 'open_earth_map_unet_lc_naip_corrected')

for RESOLUTION in "${RESOLUTIONs[@]}"; do
    for UNARY_SRC in "${UNARY_SRCs[@]}"; do
        EPSG=epsg-32611
        DATASETS=('castaic_lake' 'colorado_river')
        for DATASET in "${DATASETS[@]}"; do
            echo Running refinement for dataset: $DATASET
            python refine.py \
                --base_dir $BASE_DIR \
                --epsg $EPSG \
                --dataset $DATASET \
                --resolution $RESOLUTION \
                --unary_src $UNARY_SRC \
                --unary_filename mosaic.tiff \
                --feature_set $FEATURE_SET \
                --theta_alpha $ALPHA \
                --theta_alpha_z $ALPHA_Z \
                --theta_betas $BETA \
                --theta_gamma $GAMMA \
                --theta_gamma_z $GAMMA_Z \
                --w1 $W1 \
                --w2 $W2
        done
        EPSG=epsg-32616
        DATASETS=('kentucky_river')
        for DATASET in "${DATASETS[@]}"; do
            echo Running refinement for dataset: $DATASET
            python refine.py \
                --base_dir $BASE_DIR \
                --epsg $EPSG \
                --dataset $DATASET \
                --resolution $RESOLUTION \
                --unary_src $UNARY_SRC \
                --unary_filename mosaic.tiff \
                --feature_set $FEATURE_SET \
                --theta_alpha $ALPHA \
                --theta_alpha_z $ALPHA_Z \
                --theta_betas $BETA \
                --theta_gamma $GAMMA \
                --theta_gamma_z $GAMMA_Z \
                --w1 $W1 \
                --w2 $W2
        done

        EPSG=epsg-32618
        DATASETS=('duck')
        for DATASET in "${DATASETS[@]}"; do
            echo Running refinement for dataset: $DATASET
            python refine.py \
                --base_dir $BASE_DIR \
                --epsg $EPSG \
                --dataset $DATASET \
                --resolution $RESOLUTION \
                --unary_src $UNARY_SRC \
                --unary_filename mosaic.tiff \
                --feature_set $FEATURE_SET \
                --theta_alpha $ALPHA \
                --theta_alpha_z $ALPHA_Z \
                --theta_betas $BETA \
                --theta_gamma $GAMMA \
                --theta_gamma_z $GAMMA_Z \
                --w1 $W1 \
                --w2 $W2
        done
    done
done
