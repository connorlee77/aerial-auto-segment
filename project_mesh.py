import cv2
import numpy as np
import os
import glob

import matplotlib.pyplot as plt

import open3d as o3d


from PIL import ImageColor

HEX_COLORS = [
    '#419BDF', '#397D49', '#88B053', '#7A87C6', '#E49635', '#DFC35A',
    '#C4281B', '#ffffff', '#B39FE1', '#A8DEFF'
]

I_old = np.array([
    [511.03573247, 0.000000, 311.80346835], 
    [0.000000, 508.22913692, 261.56701122], 
    [0.000000, 0.000000, 1.000000]
])
D = np.array([-0.43339599, 0.18974767, -0.00146426, 0.00118333, 0.000000])

H, W = (512, 640)
I, roi = cv2.getOptimalNewCameraMatrix(I_old, D, (W, H), 0, (W, H))

def dynamic_world_color_map():
    rgb_colors = [ImageColor.getcolor(c, "RGB") for c in HEX_COLORS]
    color_map = dict(zip(list(range(0, 10)), rgb_colors))
    return color_map

def colorize_dynamic_world_label(label):
    mapping = dynamic_world_color_map()

    h = len(label)
    color_label = np.zeros((h, 3), dtype=np.uint8)
    for i in range(0, 10):
        color_label[label == i, :] = mapping[i] 
    return color_label / 255

pts = np.load('outputs/thermal-10000.npy').T

pts = pts[[1, 2, 0, 3], :].T


color = colorize_dynamic_world_label(pts[:,-1])

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(pts[:,:3])
pcl.colors = o3d.utility.Vector3dVector(color[:, :3])

o3d.io.write_point_cloud("10000.pcd", pcl)

pcl.estimate_normals()
normals = np.asarray(pcl.normals)
# pcl.normals = o3d.utility.Vector3dVector(-normals)

bbox = pcl.get_axis_aligned_bounding_box()

rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl)
p_mesh_crop = rec_mesh.crop(bbox)
mesh = p_mesh_crop

# o3d.visualization.draw_geometries([p_mesh_crop])
o3d.io.write_triangle_mesh('10000.ply', mesh)

img_width = 640
img_height = 512

render = o3d.visualization.rendering.OffscreenRenderer(width=img_width, height=img_height)
render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

# Define a simple unlit Material.
# (The base color does not replace the mesh's own colors.)
mtl = o3d.visualization.rendering.MaterialRecord()
mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
mtl.shader = "defaultUnlit"

render.scene.add_geometry("MyMeshModel", pcl, mtl)

render.setup_camera(I, np.eye(4), img_width, img_height)


# render.scene.camera.set_projection(I, 0.1, 1.0, 640, 512)
center = np.array([0, 0, 1])
eye = [0, 0, 0]
up = [0, 1, 0]
render.scene.camera.look_at(center, eye, up)
# render the scene with respect to the camera
img_o3d = render.render_to_image()

# we can now save the rendered image right at this point 
o3d.io.write_image("output.png", img_o3d, 9)

img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGR)
# img_cv2 = cv2.rotate(img_cv2, cv2.ROTATE_180)
cv2.imwrite("cv_output.png", img_cv2)