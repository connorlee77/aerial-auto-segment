import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2

import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt

def lookAt(eye, target, up, yz_flip=False):
    # Normalize the up vector
    up /= np.linalg.norm(up)
    forward = eye - target
    forward /= np.linalg.norm(forward)
    if np.dot(forward, up) == 1 or np.dot(forward, up) == -1:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    # Construct a rotation matrix from the right, new_up, and forward vectors
    rotation = np.eye(4)
    rotation[:3, :3] = np.row_stack((right, new_up, forward))

    # Apply a translation to the camera position
    translation = np.eye(4)
    translation[:3, 3] = [
        np.dot(right, eye),
        np.dot(new_up, eye),
        -np.dot(forward, eye),
    ]

    if yz_flip:
        # This is for different camera setting, like Open3D
        rotation[1, :] *= -1
        rotation[2, :] *= -1
        translation[1, 3] *= -1
        translation[2, 3] *= -1

    camera_pose = np.linalg.inv(np.matmul(translation, rotation))

    return camera_pose

I_old = np.array([
    [511.03573247, 0.000000, 311.80346835], 
    [0.000000, 508.22913692, 261.56701122], 
    [0.000000, 0.000000, 1.000000]
])
D = np.array([-0.43339599, 0.18974767, -0.00146426, 0.00118333, 0.000000])

H, W = (512, 640)
I, roi = cv2.getOptimalNewCameraMatrix(I_old, D, (W, H), 0, (W, H))


ply_file = '10000.ply'
fuze_trimesh = trimesh.load('10000.ply')

mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
print(mesh.centroid, mesh.bounds)
scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0, 1.0], bg_color=[0, 0, 0])

scene.add(mesh)

fx = I[0,0]
fy = I[1,1]
cx = I[0,2]
cy = I[1,2]
print(fx, fy, cx, cy)
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.1, zfar=10000)
camera_pose = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
])

eye = np.array([0, 0, -1], dtype=float)
target = np.array([0, 0, 1], dtype=float)
up = np.array([0, 1, 0], dtype=float)
camera_pose = lookAt(eye, target, up, yz_flip=False)
print(camera_pose)

scene.add(camera, pose=camera_pose)

r = pyrender.OffscreenRenderer(640, 512, point_size=1)
color, depth = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)

img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
cv2.imwrite('test.png', img)