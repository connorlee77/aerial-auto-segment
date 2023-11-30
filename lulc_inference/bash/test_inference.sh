python inference_lulc.py \
--input_path /home/connor/repos/aerial-auto-segment/planet_kentucky.tiff \
--output_path /home/connor/repos/aerial-auto-segment/test_inf_out \
--patch_size 2048 \
--stride 1920 \
--n_classes 7 \
--device 0 \
--network chesapeake-bay \
--weights_path pretrained_weights/epoch=820-step=513125.ckpt