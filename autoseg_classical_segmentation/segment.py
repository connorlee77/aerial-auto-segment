import os
import skimage
import numpy as np
import cv2
import tqdm
import glob
import argparse

def colorize_instances(mask, base_img):
    # Convert the mask to a color image
    colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in np.unique(mask):
        # random rgb
        colorized_mask[mask == i, :] = np.random.randint(0, 256, 3)

    overlay = cv2.addWeighted(base_img, 0.7, colorized_mask, 0.3, 0)
    return colorized_mask, overlay

# Segment images using classicial segmentation methods from skimage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment images using classical segmentation methods')
    parser.add_argument('--input', type=str, help='Path to the input directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--method', type=str, choices=['felzenszwalb', 'slic'], help='Segmentation method')
    args = parser.parse_args()

    # Get the input files
    input_files = sorted(glob.glob(os.path.join(args.input, '*.png')))
    assert len(input_files) > 0, 'No input files found in {}'.format(args.input)

    # Create the output directory
    os.makedirs(args.output, exist_ok=True)

    # Loop through the input files
    for input_file in tqdm.tqdm(input_files):
        # Read the input image
        img = cv2.imread(input_file, 1)

        # Segment the image
        if args.method == 'felzenszwalb':
            mask = skimage.segmentation.felzenszwalb(img, scale=1e4, sigma=0, min_size=50)
        elif args.method == 'slic':
            mask = skimage.segmentation.slic(img, n_segments=100, compactness=10, sigma=0.4)

        color_mask, overlay = colorize_instances(mask, img)

        # Save the mask
        name = os.path.basename(input_file).replace('pair-', 'thermal-')
        mask_file = os.path.join(args.output, name)
        cv2.imwrite(mask_file, mask.astype(np.int16))
        color_mask_file = os.path.join(args.output, name.replace('.png', '_color.png'))
        cv2.imwrite(color_mask_file, overlay)