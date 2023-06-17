import math
import numpy as np
from scipy.spatial.transform import Rotation

# Project world -> camera -> image
# X: [x, y, z] (3, N)
def project_points(X, R_mat, P):
    r, c = X.shape
    # Transform points from world coordinates to camera coordinate frame 
    Xr = R_mat @ X

    # Switch ONR coordinate system to opencv coordinate frame
    Xc = Xr[[1, 2, 0], :]
    Xc = np.vstack([Xc, np.ones((1, c))])

    # Project to image coordinates with camera projection matrix
    coords = P @ Xc
    
    # Homogenize points
    coords_xy = coords[0:2, :] / coords[2, :]
    return coords_xy

def estimate_horizon_mask(q_xyzw, Pn):
    N = 100
    y = 3200
    x = 5000

    r = Rotation.from_quat(q_xyzw)
    yaw, pitch, roll =  r.as_euler('ZYX', degrees=True)
    r = Rotation.from_euler('ZYX', [0, pitch, -roll], degrees=True)
    R_mat = r.as_matrix()

    xx, yy = np.meshgrid(np.linspace(0, x, N), np.linspace(-y, y, N))
    zz = np.ones_like(xx)*(100)

    X = np.stack([xx, yy, zz], axis=2)
    X = X.reshape(-1, 3).T
    Xn = project_points(X, R_mat, Pn).T

    return Xn.astype(int)

def power_spacing(num_points, start, end, exponent):
    linear_space = np.linspace(0, 1, num_points)
    power_space = np.power(linear_space, exponent)
    scaled_power_space = start + (end - start) * power_space
    return scaled_power_space

# world_pts: (N*N, 4) - x, y, z, label
def world2cam(q_xyzw, Pn, world_pts):

    r = Rotation.from_quat(q_xyzw)
    yaw, pitch, roll =  r.as_euler('ZYX', degrees=True)
    r = Rotation.from_euler('ZYX', [0, pitch, -roll], degrees=True)
    R_mat = r.as_matrix()

    X = world_pts[:, 0:3].T
    Xn = project_points(X, R_mat, Pn).T
    return Xn.astype(int)

def world_to_camera_coords(q_xyzw, world_pts):
    r = Rotation.from_quat(q_xyzw)
    yaw, pitch, roll =  r.as_euler('ZYX', degrees=True)
    r = Rotation.from_euler('ZYX', [0, pitch, -roll], degrees=True)
    R_mat = r.as_matrix()

    X = world_pts[:, 0:3].T
    # Transform points from world coordinates to camera coordinate frame 
    Xr = R_mat @ X
    return Xr

def create_world_grid(yaw, x_mag=8000, y_mag=3000, Nx=200, Ny=200, exp_x=5, exp_y=5):

        x_mag = 8000
        
        x_unit_vec = np.array([-np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)])

        # Faces right
        y_unit_vec = np.array([-np.cos(yaw+np.pi), np.sin(yaw+np.pi)])
        right_vec = y_unit_vec*y_mag

        x_space = power_spacing(Nx, 0, x_mag, exp_x)
        y_space_left = power_spacing(Ny // 2, 0.1, y_mag, exp_y)[::-1]
        y_space_right = power_spacing(Ny // 2, 0.1, y_mag, exp_y)

        x_magnitudes = x_space
        # y_magnitudes = np.concatenate((-y_space_right, y_space_left))

        # New stuff
        y_mag_start = 10
        y_magnitudes = np.linspace(-y_mag_start, y_mag_start, num=Ny).reshape(Ny, 1)
        y_lin_scale = np.linspace(1, int(math.pow(y_mag / y_mag_start, 1/3)), num=Nx).reshape(1, Nx) ** 3
        y_magnitudes_scaled = y_magnitudes * y_lin_scale

        # Create the grid of world points
        xx, yy = np.meshgrid(x_magnitudes, y_magnitudes)
        yy = y_magnitudes_scaled

        world_grid = np.stack([xx.transpose(1, 0), yy.transpose(1, 0)], axis=2)
        world_pts = world_grid.reshape(-1, 2)

        return x_unit_vec, y_unit_vec, x_magnitudes, y_magnitudes, world_pts, xx, yy