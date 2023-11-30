import warnings

import cv2
import numpy as np
import rasterio
import torch
import tqdm
import psutil
import dask.array as da
warnings.filterwarnings("ignore")


oem_class_rgb = {
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

oem_class_gray = {
    "Bareland": 0,
    "Grass": 1,
    "Pavement": 2,
    "Road": 3,
    "Tree": 4,
    "Water": 5,
    "Cropland": 6,
    "buildings": 7,
}

cb_class_rgb = {
    # 'background': (0, 0, 0),
    'water': (0, 197, 255),
    'tree canopy and shrubs': (38, 115, 0),
    'low vegetation': (163, 255, 115),
    'barren': (255, 170, 0),
    'impervious surfaces': (156, 156, 156),
    'impervious roads': (0, 0, 0),
    'aberdeen proving ground': (0, 0, 0),
}

cb_class_gray = {
    # 'background' : 0,
    'water' : 0,
    'tree canopy and shrubs' : 1,
    'low vegetation' : 2,
    'barren' : 3,
    'impervious surfaces' : 4,
    'impervious roads' : 5,
    'aberdeen proving ground' : 6,
}

def label2rgb(a, dataset="open-earth-map"):
    """
        Convert label to rgb image.
        ## Parameters
            a: np.ndarray (H, W)
                The label.
            dataset: str
                The dataset. Choices: ["open-earth-map", "chesapeake-bay"]
    """
    if dataset == "open-earth-map":
        out = np.zeros(shape=a.shape + (3,), dtype="uint8")
        for k, v in oem_class_gray.items():
            out[a == v, 0] = oem_class_rgb[k][0]
            out[a == v, 1] = oem_class_rgb[k][1]
            out[a == v, 2] = oem_class_rgb[k][2]
    elif dataset == "chesapeake-bay":
        out = np.zeros(shape=a.shape + (3,), dtype="uint8")
        for k, v in cb_class_gray.items():
            out[a == v, 0] = cb_class_rgb[k][0]
            out[a == v, 1] = cb_class_rgb[k][1]
            out[a == v, 2] = cb_class_rgb[k][2]
    
    return out


def save_img(path, img, crs, transform):
    C, H, W = img.shape
    with rasterio.open(path, 'w', driver='GTiff', height=H, width=W, count=C, dtype=img.dtype, crs=crs, transform=transform) as dst:
        dst.write(img)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def tiled_inference(network, img, n_classes, patch_size=2048, stride=1024):
    '''
        Performs tiled inference on the input image.

        ## Parameters
            network: torch.nn.Module
                The network to use for inference.
            img: np.ndarray (H, W, C)
                The input image.
            n_classes: int
                The number of classes.
            patch_size: int
                The patch size for inference.
            stride: int
                The stride for inference.
        ## Returns
            pred_label: np.ndarray (H, W)
                The predicted label.
            pred_prob: np.ndarray (n_classes, H, W)
                The predicted probability.
    '''

    if img.shape[2] > 3:
        img = img[:, :, 0:3]
    height, width, bands = img.shape

    C = int(np.ceil((width - patch_size) / stride) + 1)
    R = int(np.ceil((height - patch_size) / stride) + 1)
    # weight matrix B for avoiding boundaries of patches
    assert patch_size > stride, "patch_size {} should be larger than stride {}".format(patch_size, stride)
    w = patch_size
    s1 = stride
    s2 = w - s1
    d = 1 / (1 + s2)
    B1 = np.ones((w, w))
    B1[:, s1::] = np.dot(np.ones((w, 1)), (-np.arange(1, s2 + 1) * d + 1).reshape(1, s2))
    B2 = np.flip(B1)
    B3 = B1.T
    B4 = np.flip(B3)
    B = B1 * B2 * B3 * B4

    # B_data = B1 * B2 * B3 * B4
    # B = da.from_array(B_data, chunks=(2048, 2048))

    padded_img = np.zeros((patch_size + stride * (R - 1), patch_size + stride * (C - 1), 3), dtype=np.uint8)
    ph, pw, pc = padded_img.shape
    padded_img[0:height, 0:width, :] = img
    padded_img[height:, :, :] = cv2.flip(padded_img[height - (ph - height):height, :, :], 0)
    padded_img[:, width:, :] = cv2.flip(padded_img[:, width - (pw - width):width, :], 1)

    # cv2.imwrite('padded_img.png', cv2.cvtColor(padded_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    # print('saved...')
    weight = np.zeros((patch_size + stride * (R - 1), patch_size + stride * (C - 1)), dtype=np.float32)

    pred_prob = np.memmap('pred_prob.dat', dtype=np.float32, mode='w+', shape=(n_classes + 1, patch_size + stride * (R - 1), patch_size + stride * (C - 1)))
    pred_prob[:] = np.zeros((n_classes + 1, patch_size + stride * (R - 1), patch_size + stride * (C - 1)), dtype=np.float32)
    with tqdm.tqdm(total=R*C, desc="Inference") as pbar:
        for r in range(R):
            for c in range(C):
                img_tile = padded_img[r * stride:r * stride + patch_size, c *
                                    stride:c * stride + patch_size, :].astype(np.float32) / 255
                
                # img_tile = padded_img[r * stride:r * stride + patch_size, c *
                #                     stride:c * stride + patch_size, :]
                # img_tile = change_brightness(img_tile, value=80)
                # img_tile = img_tile.astype(np.float32) / 255

                # Create batch of augmented images for TTA.
                imgs = []
                imgs.append(img_tile.copy())
                imgs.append(img_tile[:, ::-1, :].copy())
                imgs.append(img_tile[::-1, :, :].copy())
                imgs.append(img_tile[::-1, ::-1, :].copy())

                input = torch.cat([torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
                                for x in imgs], dim=0).float().to(network.device)

                with torch.no_grad():
                    mask = network(input)
                    mask = torch.softmax(mask, dim=1)
                    mask = mask.cpu().numpy()

                    # Recombining test time augmentations via average.
                    pred = (mask[0, :, :, :] + mask[1, :, :, ::-1] + mask[2, :, ::-1, :] + mask[3, :, ::-1, ::-1]) / 4

                pred_prob[0:n_classes, r * stride:r * stride + patch_size, c * stride:c * stride + patch_size] += pred * B
                weight[r * stride:r * stride + patch_size, c * stride:c * stride + patch_size] += B
    
                pbar.update(1)
    print(weight.shape, pred_prob.shape, weight.dtype, pred_prob.dtype)
    for b in range(n_classes):
        pred_prob[b, :, :] /= weight
    pred_prob[-1, :, :] = pred_prob[0:n_classes, :, :].argmax(axis=0)
    return pred_prob[:, 0:height, 0:width]
