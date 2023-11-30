PLACE=$1
echo Performing inference on ${PLACE}...

# echo Performing inference on ${PLACE}...
# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/naip/${PLACE}/mosaic/mosaic.tiff \
# --output_path /data/microsoft_planetary_computer/open_earth_map_unet_lc_naip/${PLACE}/mosaic \
# --patch_size 2048 \
# --stride 1948 \
# --n_classes 8 \
# --device 0 \
# --network open-earth-map

# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/planet/${PLACE}/mosaic/mosaic.tiff \
# --output_path /data/microsoft_planetary_computer/open_earth_map_unet_lc_planet/${PLACE}/mosaic \
# --patch_size 2048 \
# --stride 512 \
# --n_classes 8 \
# --device 0 \
# --network open-earth-map

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/naip/${PLACE}/mosaic/mosaic.tiff \
--output_path /data/microsoft_planetary_computer/chesapeake_bay_swin_crossentropy_lc_naip/${PLACE}/mosaic \
--patch_size 512 \
--stride 256 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--weights_path pretrained_weights/swin_cb_ce.ckpt

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/planet/${PLACE}/mosaic/mosaic.tiff \
--output_path /data/microsoft_planetary_computer/chesapeake_bay_swin_crossentropy_lc_planet/${PLACE}/mosaic \
--patch_size 512 \
--stride 256 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--weights_path pretrained_weights/swin_cb_ce.ckpt


python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/naip/${PLACE}/mosaic/mosaic.tiff \
--output_path /data/microsoft_planetary_computer/chesapeake_bay_swin_focalloss_lc_naip/${PLACE}/mosaic \
--patch_size 512 \
--stride 256 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--weights_path pretrained_weights/swin_cb_fl.ckpt

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/planet/${PLACE}/mosaic/mosaic.tiff \
--output_path /data/microsoft_planetary_computer/chesapeake_bay_swin_focalloss_lc_planet/${PLACE}/mosaic \
--patch_size 512 \
--stride 256 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--weights_path pretrained_weights/swin_cb_fl.ckpt

