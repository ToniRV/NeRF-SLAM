#!/usr/bin/env python3

import cv2

import numpy as np
from scipy.spatial.transform import Rotation

from icecream import ic
import torch

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    try:
        v = np.cross(a, b)
    except:
        pass
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    try:
        c = np.cross(da, db)
    except:
        pass
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(image):
    if image is str:
        image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def pose_matrix_to_t_and_quat(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], q], axis=0)

def get_pose_from_df(gt_df):
    x = gt_df.loc['tx'] # [m]
    y = gt_df.loc['ty'] # [m]
    z = gt_df.loc['tz'] # [m]
    qx = gt_df.loc['qx']
    qy = gt_df.loc['qy']
    qz = gt_df.loc['qz']
    qw = gt_df.loc['qw']
    r = Rotation.from_quat([qx, qy, qz, qw])
    pose = np.eye(4)
    pose[:3, :3] = r.as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose

def get_velocity(euroc_df):
    # TODO: rename these
    vx = euroc_df.loc[' v_RS_R_x [m s^-1]']
    vy = euroc_df.loc[' v_RS_R_y [m s^-1]']
    vz = euroc_df.loc[' v_RS_R_z [m s^-1]']
    return np.array([vx, vy, vz])

def get_bias(euroc_df):
    # TODO: rename these
    ba_x = euroc_df.loc[' b_a_RS_S_x [m s^-2]']
    ba_y = euroc_df.loc[' b_a_RS_S_y [m s^-2]']
    ba_z = euroc_df.loc[' b_a_RS_S_z [m s^-2]']
    bg_x = euroc_df.loc[' b_w_RS_S_x [rad s^-1]']
    bg_y = euroc_df.loc[' b_w_RS_S_y [rad s^-1]']
    bg_z = euroc_df.loc[' b_w_RS_S_z [rad s^-1]']
    return np.array([ba_x, ba_y, ba_z, bg_x, bg_y, bg_z])


# offse's default is 0.5 bcs we scale/offset poses to 1.0/0.5 before feeding to nerf
def nerf_matrix_to_ngp(nerf_matrix, scale=1.0, offset=0.5): 
    result = nerf_matrix.copy()
    result[:3, 1] *= -1
    result[:3, 2] *= -1
    result[:3, 3] = result[:3, 3] * scale + offset

    # Cycle axes xyz<-yzx
    tmp = result[0, :].copy()
    result[0, :] = result[1, :]
    result[1, :] = result[2, :]
    result[2, :] = tmp

    return result

# offset's default is 0.5 bcs we scale/offset poses to 1.0/0.5 before feeding to nerf
def ngp_matrix_to_nerf(ngp_matrix, scale=1.0, offset=0.5):
    result = ngp_matrix.copy()

    # Cycle axes xyz->yzx
    tmp = result[2, :].copy()
    result[1, :] = result[0, :]
    result[2, :] = result[1, :]
    result[0, :] = tmp

    result[:3, 0] *=  1 / scale
    result[:3, 1] *= -1 / scale
    result[:3, 2] *= -1 / scale
    result[:3, 3] = (result[:3, 3] - offset) / scale

    return result;

# This can be very slow, send to gpu device...
def srgb_to_linear(img, device):
    img = img.to(device=device)
    limit = 0.04045
    return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def get_scale_and_offset(aabb):
    # map the given aabb of the form [[minx,miny,minz],[maxx,maxy,maxz]]
    # via an isotropic scale and translate to fit in the (0,0,0)-(1,1,1) cube,
    # with the given center at 0.5,0.5,0.5
    aabb = np.array(aabb, dtype=np.float64)
    dx = aabb[1][0]-aabb[0][0]
    dy = aabb[1][1]-aabb[0][1]
    dz = aabb[1][2]-aabb[0][2]
    length = max(0.000001, max(max(abs(dx), abs(dy)), abs(dz)))
    scale = 1.0 / length
    offset = np.array([((aabb[1][0]+aabb[0][0])*0.5) * -scale + 0.5,
                       ((aabb[1][1]+aabb[0][1])*0.5) * -scale + 0.5,
                       ((aabb[1][2]+aabb[0][2])*0.5) * -scale + 0.5])
    return scale, offset


def scale_offset_poses(poses, scale, offset):  # for c2w poses!
    poses[:, :3, 3] = poses[:, :3, 3] * scale + offset
    return poses


def mse2psnr(x):
    return -10.*np.log(x)/np.log(10.)


def L2(img, ref):
    return (img - ref)**2


def compute_mse_img(img, ref):
    img[np.logical_not(np.isfinite(img))] = 0
    img = np.maximum(img, 0.)
    return L2(img, ref)


def compute_error(img, ref):
    metric_map = compute_mse_img(img, ref)
    metric_map[np.logical_not(np.isfinite(metric_map))] = 0
    if len(metric_map.shape) == 3:
        metric_map = np.mean(metric_map, axis=2)
    mean = np.mean(metric_map)
    return mean
