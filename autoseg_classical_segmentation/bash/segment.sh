method=$1

sets=(
    2022-05-15_ColoradoRiver/flight2
    2022-05-15_ColoradoRiver/flight3
    2022-05-15_ColoradoRiver/flight4
    2022-12-20_Castaic_Lake/flight4
    2023-03-XX_Duck/ONR_2023-03-21-09-59-39
    2023-03-XX_Duck/ONR_2023-03-21-14-06-04
    # 2023-03-XX_Duck/ONR_2023-03-21-18-20-21
    2023-03-XX_Duck/ONR_2023-03-21-19-55-11
    # 2023-03-XX_Duck/ONR_2023-03-22-14-31-06
    2023-03-XX_Duck/ONR_2023-03-22-14-41-46
    2021-09-09-KentuckyRiver/flight1-1
    2021-09-09-KentuckyRiver/flight2-1
    2021-09-09-KentuckyRiver/flight3-1
)

for set in ${sets[@]}; do
(
    python segment.py \
    --input /data/onr-thermal/cogito-annotation/converted-cogito-annotations-final/${set}/thermal8/ \
    --output ${method}/${set} \
    --method ${method}
) &
done
wait
# taskset --cpu-list 50-65 bash bash/segment.sh felzenszwalb
# taskset --cpu-list 66-80 bash bash/segment.sh slic