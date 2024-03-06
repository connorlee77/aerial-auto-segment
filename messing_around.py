import startinpy
import rasterio
import numpy as np
import time

d3_path = '/data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/castaic_lake/dem/1.0/mosaic.tiff'
lulc_path = '/data/microsoft_planetary_computer/outputs/preprocessed/epsg-32611/castaic_lake/dynamicworld/1.0/mosaic.tiff'

x = np.zeros((256000000, 3))
y = np.random.rand(3, 3)

t1 = time.time()
x@y
t2 = time.time()
print(t2 - t1)

# with rasterio.open(d3_path, 'r') as src:
#     dsm_array = src.read(1)

#     height, width = dsm_array.shape
#     cols, rows = np.meshgrid(np.arange(width), np.arange(height))
#     xs, ys = rasterio.transform.xy(src.transform, rows, cols)
#     dsm_utm_grid = np.stack([xs, ys], axis=2).reshape(-1, 2)

# # with rasterio.open(lulc_path, 'r') as src:

# #     label_array = src.read()
# #     n_bands, height, width = label_array.shape
# #     cols, rows = np.meshgrid(np.arange(width), np.arange(height))
# #     xs, ys = rasterio.transform.xy(src.transform, rows, cols)
# #     label_utm_grid = np.stack([xs, ys], axis=2).reshape(-1, 2)

# # print(dsm_array.shape, label_array.shape)

# print(dsm_utm_grid.shape, dsm_array.reshape(-1, 1).shape)
# pts = np.concatenate([dsm_utm_grid, dsm_array.reshape(-1, 1)], axis=1)
# print(dsm_utm_grid.shape, dsm_array.reshape(-1, 1).shape, pts.shape)
# t1 = time.time()
# dt = startinpy.DT()
# xyz = pts
# dt.insert(xyz, insertionstrategy="BBox")
# t2 = time.time()
# print(t2 - t1)
# vertices = dt.points[1:]

# # startinpy returns vertices in (x, y, z) order, but we need (y, z, x) for rendering in our coordinate frame.
# vertices = vertices[:, [1, 2, 0]]