EXP_NAME=$1
ALPHA=$2
BETA=$3
GAMMA=$4
W1=$5
W2=$6

echo Running experiment: $EXP_NAME

echo Parameters:
echo theta_alpha: $ALPHA
echo theta_beta: $BETA
echo theta_gamma: $GAMMA
echo w1: $W1
echo w2: $W2
echo

# Parameters that shouldn't change for chesapeake bay datasets
BASE_DIR=/media/hdd2/data/chesapeake_bay_lulc/outputs/preprocessed
RESOLUTION=1.0
UNARY_SRC=dynamicworld
FEATURE_SET='naip'

EPSG=epsg-32618
DATASETS=('virginia_beach_creeds' 'clinton' 'virginia_beach_false_cape_landing')

for DATASET in "${DATASETS[@]}"
do  
    echo Running refinement for dataset: $DATASET
    python refine.py \
    --base_dir $BASE_DIR \
    --epsg $EPSG \
    --dataset $DATASET \
    --resolution $RESOLUTION \
    --unary_src $UNARY_SRC \
    --feature_set $FEATURE_SET \
    --theta_alpha $ALPHA \
    --theta_beta $BETA \
    --theta_gamma $GAMMA \
    --w1 $W1 \
    --w2 $W2 \
    --output_dir lulc_refine_outputs/${EXP_NAME}/${EPSG}/${DATASET}/refined_lulc/${RESOLUTION}/${UNARY_SRC}/${FEATURE_SET// /_}
done