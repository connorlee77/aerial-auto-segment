# NAIP
python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32616/kentucky_river/naip/1.0/mosaic.tiff \
--output_path /home/connor/repos/aerial-auto-segment/kentucky \
--patch_size 2048 \
--stride 1984 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--in_channels 4 \
--mean 0.43206426 0.5120306 0.46068674 0.6469401 \
--std 0.1537372 0.1355305 0.09609849 0.15973314 \
--weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/castaic_lake/naip/1.0/mosaic.tiff \
--output_path /home/connor/repos/aerial-auto-segment/castaic_lake \
--patch_size 2048 \
--stride 1984 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--in_channels 4 \
--mean 0.43206426 0.5120306 0.46068674 0.6469401 \
--std 0.1537372 0.1355305 0.09609849 0.15973314 \
--weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32618/duck/naip/1.0/mosaic.tiff \
--output_path /home/connor/repos/aerial-auto-segment/duck \
--patch_size 2048 \
--stride 1984 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--in_channels 4 \
--mean 0.43206426 0.5120306 0.46068674 0.6469401 \
--std 0.1537372 0.1355305 0.09609849 0.15973314 \
--weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/big_bear_lake/naip/1.0/mosaic.tiff \
--output_path /home/connor/repos/aerial-auto-segment/big_bear_lake \
--patch_size 2048 \
--stride 1984 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--in_channels 4 \
--mean 0.43206426 0.5120306 0.46068674 0.6469401 \
--std 0.1537372 0.1355305 0.09609849 0.15973314 \
--weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

python inference_lulc.py \
--input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/colorado_river/naip/1.0/mosaic.tiff \
--output_path /home/connor/repos/aerial-auto-segment/colorado_river \
--patch_size 2048 \
--stride 1984 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--in_channels 4 \
--mean 0.43206426 0.5120306 0.46068674 0.6469401 \
--std 0.1537372 0.1355305 0.09609849 0.15973314 \
--weights_path pretrained_weights/good_weights/jtom5v7t.ckpt


# # Planet
# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32616/kentucky_river/planet/1.0/mosaic.tiff \
# --output_path /home/connor/repos/aerial-auto-segment/kentucky \
# --patch_size 2048 \
# --stride 1984 \
# --n_classes 7 \
# --device 0 \
# --network chesapeake-bay \
# --in_channels 4 \
# --mean 0.22637588 0.28828233 0.27453664 0.33777460958805067 \
# --std 0.10067633 0.09112308 0.07367223 0.10437626590039228 \
# --weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/castaic_lake/planet/1.0/mosaic.tiff \
# --output_path /home/connor/repos/aerial-auto-segment/castaic_lake \
# --patch_size 2048 \
# --stride 1984 \
# --n_classes 7 \
# --device 0 \
# --network chesapeake-bay \
# --in_channels 4 \
# --mean 0.3623688185908874 0.3252184599733136 0.3156119953488237 0.20310992114315574 \
# --std 0.17425752152591054 0.13928294880583034 0.11524457889477047 0.08703511384815744 \
# --weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32618/duck/planet/1.0/mosaic.tiff \
# --output_path /home/connor/repos/aerial-auto-segment/duck \
# --patch_size 2048 \
# --stride 1984 \
# --n_classes 7 \
# --device 0 \
# --network chesapeake-bay \
# --in_channels 4 \
# --mean 0.3623688185908874 0.3252184599733136 0.3156119953488237 0.20310992114315574 \
# --std 0.17425752152591054 0.13928294880583034 0.11524457889477047 0.08703511384815744 \
# --weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/big_bear_lake/planet/1.0/mosaic.tiff \
# --output_path /home/connor/repos/aerial-auto-segment/big_bear_lake \
# --patch_size 2048 \
# --stride 1984 \
# --n_classes 7 \
# --device 0 \
# --network chesapeake-bay \
# --in_channels 4 \
# --mean 0.3623688185908874 0.3252184599733136 0.3156119953488237 0.20310992114315574 \
# --std 0.17425752152591054 0.13928294880583034 0.11524457889477047 0.08703511384815744 \
# --weights_path pretrained_weights/good_weights/jtom5v7t.ckpt

# python inference_lulc.py \
# --input_path /data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/colorado_river/planet/1.0/mosaic.tiff \
# --output_path /home/connor/repos/aerial-auto-segment/colorado_river \
# --patch_size 2048 \
# --stride 1984 \
# --n_classes 7 \
# --device 0 \
# --network chesapeake-bay \
# --in_channels 4 \
# --mean 0.3623688185908874 0.3252184599733136 0.3156119953488237 0.20310992114315574 \
# --std 0.17425752152591054 0.13928294880583034 0.11524457889477047 0.08703511384815744 \
# --weights_path pretrained_weights/good_weights/jtom5v7t.ckpt