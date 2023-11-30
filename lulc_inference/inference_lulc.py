import argparse
import os
import warnings
import cv2
import numpy as np
import rasterio
import torch
from PIL import Image

warnings.filterwarnings("ignore")

from models.openearthmap import OpenEarthMapNetwork
from models.chesapeakebay_swin import ChesapeakeBaySwinNetwork
from utils.utils import label2rgb, save_img, tiled_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="/data/chesapeake_bay_lulc/outputs/preprocessed/epsg-32618/virginia_beach_creeds/naip/1.0/mosaic.tiff")
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--patch_size", type=int, default=2048, help="patch size for inference")
    parser.add_argument("--stride", type=int, default=1024, help="stride for inference")
    parser.add_argument("--n_classes", type=int, default=8, help="number of classes")
    parser.add_argument("--device", type=str, default="0", help="gpu id")
    parser.add_argument("--network", type=str, default="open-earth-map",
                        help="type of pretrained network to use", choices=["open-earth-map", "chesapeake-bay"])
    parser.add_argument("--weights_path", type=str, default="weights/chesapeake_bay_swinformer.pth",
                        help="path to pretrained weights")
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() else "cpu"
    if args.network == "chesapeake-bay":
        network = ChesapeakeBaySwinNetwork(device, args.weights_path, args.patch_size)
    elif args.network == "open-earth-map":
        print('Pretrained weights path is fixed to pretrained_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth')
        network = OpenEarthMapNetwork(device, weights_path='pretrained_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth')

    img, crs, trans = None, None, None
    with rasterio.open(args.input_path, "r") as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        crs, trans = src.crs, src.transform

    H, W, C = img.shape
    pred_prob = tiled_inference(network, img, args.n_classes,
                                            patch_size=args.patch_size, stride=args.stride)
    assert pred_prob.shape == (args.n_classes + 1, H, W)

    print('Done tiling')
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    preview_path = os.path.join(args.output_path, "mosaic_preview.png")
    probability_lc_path = os.path.join(args.output_path, "mosaic.tiff")
    pred_path = os.path.join(args.output_path, "mosaic_labels.tiff")

    # save png image
    pr_rgb = label2rgb(pred_prob[-1, :, :], dataset=args.network)
    resize_rgb = cv2.resize(pr_rgb, (0,0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(resize_rgb).save(preview_path)
    print('Done saving rgb')
    # save rgb geotiff
    # pr_rgb = np.transpose(pr_rgb, (2, 0, 1))
    # save_img(pred_path, pr_rgb, crs, trans)
    # print('Done saving rgb geotiff')
    # save probability map + labels
    save_img(probability_lc_path, pred_prob, crs, trans)
    print('Done saving rgb prob geotiff')
