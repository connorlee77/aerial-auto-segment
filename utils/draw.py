import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
import shutil
import matplotlib.pyplot as plt
from horizon.flir_boson_settings import I, D, P

import tqdm
import quaternion
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.special import softmax

import rasterio
import rasterio.plot
import rasterio.merge
import rasterio.mask
import pyproj
from PIL import ImageColor
import shapely
from shapely import Polygon
import skimage


def dynamic_world_color_map():
    HEX_COLORS = [
        '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635', '#DFC35A',
        '#C4281B', '#ffffff', '#B39FE1'
    ]

    # HEX_COLOR_MAPPING = {
    #     '#419BDF': 0,  # water
    #     '#397D49': 1,  # trees
    #     '#88B053': 2,  # grass
    #     '#7A87C6': 3,  # Flooded vegetation
    #     '#E49635': 4,  # Crops
    #     '#DFC35A': 5,  # Shrub and scrub
    #     '#C4281B': 6,  # built
    #     '#ffffff': 7,  # bare
    #     '#B39FE1': 8,  # snow and ice
    # }
    rgb_colors = [ImageColor.getcolor(c, "RGB") for c in HEX_COLORS]
    color_map = dict(zip(list(range(0, 9)), rgb_colors))
    return color_map


def chesapeake_landcover_color_map():
    # name_map = {
    #     1: 'Water',
    #     2: 'Emergent Wetlands',
    #     3: 'Tree Canopy',
    #     4: 'Scrub\Shrub',
    #     5: 'Low Vegetation',
    #     6: 'Barren',
    #     7: 'Impervious Structures',
    #     8: 'Other Impervious',
    #     9: 'Impervious Roads',
    #     10: 'Tree Canopy over Impervious Structures',
    #     11: 'Tree Canopy over Other Impervious',
    #     12: 'Tree Canopy over Impervious Roads',
    # }

    color_map = {
        1: np.array([0,  97, 255], dtype=np.uint8),
        2: np.array([0, 168, 132], dtype=np.uint8),
        3: np.array([38, 115,   0], dtype=np.uint8),
        4: np.array([76, 230,   0], dtype=np.uint8),
        5: np.array([165, 245, 122], dtype=np.uint8),
        6: np.array([255, 170,   0], dtype=np.uint8),
        7: np.array([255,   0,   0], dtype=np.uint8),
        8: np.array([178, 178, 178], dtype=np.uint8),
        9: np.array([0, 0, 0], dtype=np.uint8),
        10: np.array([115, 115,   0], dtype=np.uint8),
        11: np.array([205, 205, 102], dtype=np.uint8),
        12: np.array([255, 255, 115], dtype=np.uint8),
    }

    return color_map


def common_color_map():
    common_colors = {
        0: '#419BDF',  # water
        1: '#397D49',  # trees, tree canopy, tree canopy over impervious structures, tree canopy over other impervious, tree canopy over impervious roads
        2: '#88B053',  # grass, crops, low vegetation
        3: '#DFC35A',  # scrub and shrub
        4: '#7A87C6',  # flooded vegetation, emergent wetlands
        5: '#C4281B',  # built, impervious structures, other impervious, impervious roads
        6: '#ffffff',  # bare, barren
        7: '#B39FE1',  # snow and ice
    }

    color_map = {}
    for idx, color in common_colors.items():
        color_map[idx] = ImageColor.getcolor(color, "RGB")
    return color_map


def generate_binary_mask(img_shape, points, blur_size=5, blur_sigma=3, morph_kernel_size=5, iterations=1):
    binary_mask = np.zeros(img_shape, dtype=np.uint8)

    # Sort points by their x values
    sorted_points = sorted(points, key=lambda pt: pt[0])

    # Create a list of points for the polygon that covers the area above the given points
    polygon_points = [(0, 0)] + sorted_points + [(img_shape[1] - 1, 0)]

    # Convert the list of points to the format required by cv2.fillPoly
    polygon_points_array = np.array(
        polygon_points, dtype=np.int32).reshape(-1, 1, 2)

    # Fill the polygon
    cv2.fillPoly(binary_mask, [polygon_points_array], 1)

    # Apply Gaussian blur to the mask
    binary_mask = cv2.GaussianBlur(binary_mask.astype(
        np.float32), (blur_size, blur_size), blur_sigma)

    # Threshold the mask to create a smoother transition between the masked and unmasked regions
    _, binary_mask = cv2.threshold(binary_mask, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological operations (dilation followed by erosion)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=iterations)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=iterations)

    return binary_mask.astype(bool)


def points_to_segmentation(img_shape, points, labels, color_map, mask=None):
    x, y = np.indices(img_shape)
    xy = np.column_stack((y.ravel(), x.ravel()))

    # Create KDTree from the points
    tree = KDTree(points)

    # Find nearest neighbors for each pixel in the image
    _, nearest_neighbors = tree.query(xy)

    # Create segmentation mask using the labels of the nearest neighbors
    segmentation_labels = labels[nearest_neighbors].reshape(img_shape)

    # Apply the binary mask if provided
    if mask is not None:
        segmentation_labels = np.where(mask, segmentation_labels, -1)

    # Create colorized segmentation mask
    colorized_mask = np.zeros((*img_shape, 3), dtype=np.uint8)
    for label, color in color_map.items():
        colorized_mask[segmentation_labels == label] = color

    return colorized_mask


def points_to_surface(img_shape, points, surface, mask=None):
    H, W = img_shape
    x, y = np.indices(img_shape)
    xy = np.column_stack((y.ravel(), x.ravel()))

    tree = KDTree(points)
    _, nearest_neighbors = tree.query(xy)
    surface_img = surface[nearest_neighbors].reshape(H, W)

    surface_img = np.nan_to_num(surface_img, 0)
    if mask is not None:
        surface_img[~mask] = 200

    return surface_img


def project_elevation(img_shape, points, surface):
    H, W = img_shape

    idx_valid_w = np.logical_and(
        np.abs(points[:, 0]) >= 0, np.abs(points[:, 0]) < W)
    idx_valid_h = np.logical_and(
        np.abs(points[:, 1]) >= 0, np.abs(points[:, 1]) < H)
    idx_valid = np.logical_and(idx_valid_w, idx_valid_h)
    points = points[idx_valid]
    surface = surface[idx_valid]

    # Generate a binary mask from the original points
    binary_mask = generate_binary_mask(img_shape, points)
    probability_img = points_to_surface(
        img_shape, points, surface, mask=binary_mask)
    probability_img = cv2.rotate(probability_img, cv2.ROTATE_180)
    return probability_img


def points_to_prob(img_shape, points, prob, mask=None):
    _, C = prob.shape
    H, W = img_shape
    x, y = np.indices(img_shape)
    xy = np.column_stack((y.ravel(), x.ravel()))

    tree = KDTree(points)
    _, nearest_neighbors = tree.query(xy)
    probablities = prob[nearest_neighbors, :].reshape(H, W, C)

    probablities = np.nan_to_num(probablities, 0)
    probablities[:, :, -1] = 0  # use label channel as sky
    if mask is not None:
        probablities[~mask, :] = 0
        probablities[~mask, -1] = 0.99

    # probablities = softmax(probablities, axis=2)
    return probablities


def project_prob(img_shape, points, prob):
    H, W = img_shape

    idx_valid_w = np.logical_and(
        np.abs(points[:, 0]) >= 0, np.abs(points[:, 0]) < W)
    idx_valid_h = np.logical_and(
        np.abs(points[:, 1]) >= 0, np.abs(points[:, 1]) < H)
    idx_valid = np.logical_and(idx_valid_w, idx_valid_h)
    points = points[idx_valid]
    prob = prob[idx_valid]

    # Generate a binary mask from the original points
    binary_mask = generate_binary_mask(img_shape, points)
    probability_img = points_to_prob(img_shape, points, prob, mask=binary_mask)
    probability_img = cv2.rotate(probability_img, cv2.ROTATE_180)
    return probability_img


def draw_overlay_and_labels(img, points, labels, color_map=None, alpha=0.5):
    orig_img = np.copy(img)
    img_shape = img.shape[:2]

    w = img_shape[1]
    h = img_shape[0]

    idx_valid_w = np.logical_and(
        np.abs(points[:, 0]) >= 0, np.abs(points[:, 0]) < w)
    idx_valid_h = np.logical_and(
        np.abs(points[:, 1]) >= 0, np.abs(points[:, 1]) < h)
    idx_valid = np.logical_and(idx_valid_w, idx_valid_h)
    points = points[idx_valid]
    labels = labels[idx_valid]

    # Generate a binary mask from the original points
    binary_mask = generate_binary_mask(img_shape, points)
    colorized_mask = points_to_segmentation(
        img_shape, points, labels, color_map, mask=binary_mask)

    # Overlay the colorized segmentation mask on the image
    overlay = cv2.addWeighted(orig_img, 1 - alpha, colorized_mask, alpha, 0)
    overlay = cv2.rotate(overlay, cv2.ROTATE_180)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    img_with_pts = np.copy(img)
    # img_with_pts = np.zeros_like(img)
    for i in range(points.shape[0]):
        # print(points[i], labels[i])
        xi, yi = points[i]
        cls = labels[i]
        if not np.isnan(cls):
            color = color_map[cls]
            cv2.circle(img_with_pts, (xi, yi), radius=1,
                       color=color, thickness=-1)

    img_with_pts = cv2.rotate(img_with_pts, cv2.ROTATE_180)
    img_with_pts = cv2.cvtColor(img_with_pts, cv2.COLOR_RGB2BGR)

    orig_img = cv2.rotate(orig_img, cv2.ROTATE_180)
    return orig_img, overlay, img_with_pts


def colorize_dynamic_world_label(label):
    mapping = dynamic_world_color_map()

    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, 9): # labels 0-8 inclusive
        color_label[label == i, :] = mapping[i]

    return color_label


def colorize_chesapeake_landcover_label(label):
    mapping = chesapeake_landcover_color_map()

    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, 13):  # labels 1-12 inclusive
        color_label[label == i, :] = mapping[i]

    return color_label

def colorize_common_landcover_label(label):
    mapping = common_color_map()

    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, 8):  # labels 0-7 inclusive
        color_label[label == i, :] = mapping[i]

    return color_label


def write_text_img(image, text, output_path):
    cv2.putText(
        image,
        text=text,
        org=(10, 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2, color=(0, 0, 255), thickness=2,
        bottomLeftOrigin=True
    )
    cv2.imwrite(output_path, image)
