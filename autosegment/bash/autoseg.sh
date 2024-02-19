# d3_dtype=dem
# resolution=1.0
# lulc_type=dynamicworld

resolution=$1
d3_type_list=('dem' 'dsm' 'dem_1m')
lulc_types=(
    'dynamicworld'
    'chesapeake_bay_swin_crossentropy_lc_naip_corrected'
    'chesapeake_bay_swin_crossentropy_lc_planet'
    'open_earth_map_unet_lc_naip_corrected'
    'open_earth_map_unet_lc_planet'
)
refinement_type_list=(
    'none'
    'crf_naip_naip-nir'
    'crf_naip_naip-nir_surface_height'
    'crf_planet'
    'crf_planet_surface_height'
)

for refinement_type in ${refinement_type_list[@]}; do
    for d3_dtype in ${d3_type_list[@]}; do
        for lulc_type in ${lulc_types[@]}; do

            ### Dont change below this line unless adding new datasets

            # Duck
            dataset=2023-03-XX_Duck
            epsg=epsg-32618
            place=duck
            trajectories=(
                # ONR_2023-03-21-18-20-21 # something is wrong with this trajectory
                ONR_2023-03-22-14-41-46
                # ONR_2023-03-22-08-44-31 # no aligned.csv
                ONR_2023-03-21-19-55-11
                ONR_2023-03-21-14-06-04
                ONR_2023-03-21-09-59-39
            )

            for trajectory in ${trajectories[@]}; do
                bash bash/autoseg_helper.sh $dataset $trajectory $place $d3_dtype $epsg $lulc_type $resolution $refinement_type
            done

            # Castaic Lake
            dataset=2022-12-20_Castaic_Lake
            epsg=epsg-32611
            place=castaic_lake
            trajectories=('flight4')
            for trajectory in ${trajectories[@]}; do
                bash bash/autoseg_helper.sh $dataset $trajectory $place $d3_dtype $epsg $lulc_type $resolution $refinement_type
            done

            # Colorado River
            dataset=2022-05-15_ColoradoRiver
            epsg=epsg-32611
            place=colorado_river
            trajectories=(
                flight2
                flight3
                flight4
            )
            for trajectory in ${trajectories[@]}; do
                bash bash/autoseg_helper.sh $dataset $trajectory $place $d3_dtype $epsg $lulc_type $resolution $refinement_type
            done

            # Kentucky River
            dataset=2021-09-09-KentuckyRiver
            epsg=epsg-32616
            place=kentucky_river
            trajectories=(
                flight1-1
                flight2-1
                flight3-1
            )
            for trajectory in ${trajectories[@]}; do
                bash bash/autoseg_helper.sh $dataset $trajectory $place $d3_dtype $epsg $lulc_type $resolution $refinement_type
            done

        done
    done
done
