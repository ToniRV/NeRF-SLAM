#!/usr/bin/env python3

import numpy as np
import open3d as o3d

from abc import abstractclassmethod
from torch.utils.data.dataset import Dataset

class Dataset(Dataset):
    def __init__(self, name, args, device) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self.device = device

        self.dataset_dir = args.dataset_dir
        self.initial_k   = args.initial_k # first frame to load
        self.final_k     = args.final_k # last frame to load, if -1 load all
        self.img_stride  = args.img_stride # stride for loading images
        self.stereo      = args.stereo

        self.viz = False

        # list of data packets,
        # each data packet consists of two frames and all imu data in between
        self.data_packets = None 

    @abstractclassmethod
    def stream(self):
        pass

class PointCloudTransmissionFormat:
    def __init__(self, pointcloud: o3d.geometry.PointCloud):
        self.points = np.array(pointcloud.points)
        self.colors = np.array(pointcloud.colors)
        self.normals = np.array(pointcloud.normals)

    def create_pointcloud(self) -> o3d.geometry.PointCloud:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.points)
        pointcloud.colors = o3d.utility.Vector3dVector(self.colors)
        pointcloud.normals = o3d.utility.Vector3dVector(self.normals)
        return pointcloud

class CameraModel:
    def __init__(self, model) -> None:
        self.model = model
        pass

    def project(self, xyz):
        return self.model.project(xyz)

    def backproject(self, uv):
        return self.model.backproject(uv)

class Resolution:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
    
    def numpy(self):
        return np.array([self.width, self.height])

    def total(self):
        return self.width * self.height

class PinholeCameraModel(CameraModel):
    def __init__(self, fx, fy, cx, cy) -> None:
        super().__init__('Pinhole')
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.K = np.eye(3)
        self.K[0,0] = fx
        self.K[0,2] = cx
        self.K[1,1] = fy
        self.K[1,2] = cy

    def scale_intrinsics(self, scale_x, scale_y):
        self.fx *= scale_x # fx, cx
        self.cx *= scale_x # fx, cx
        self.fy *= scale_y # fx, cx
        self.cy *= scale_y # fx, cx

        self.K = np.eye(3)
        self.K[0,0] = self.fx
        self.K[0,2] = self.cx
        self.K[1,1] = self.fy
        self.K[1,2] = self.cy

    def numpy(self):
        return np.array([self.fx, self.fy, self.cx, self.cy])

    def matrix(self):
        return self.K

class DistortionModel:
    def __init__(self, model) -> None:
        self.model = model

class RadTanDistortionModel(DistortionModel):
    def __init__(self, k1, k2, p1, p2) -> None:
        super().__init__('RadTan')
        # Distortioncoefficients=(k1 k2 p1 p2 k3)  #OpenCV convention
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def get_distortion_as_vector(self):
        return np.array([self.k1, self.k2, self.p1, self.p2])

class CameraCalibration:
    def __init__(self, body_T_cam, camera_model, distortion_model, rate_hz, resolution, aabb, depth_scale) -> None:
        self.body_T_cam = body_T_cam
        self.camera_model = camera_model
        self.distortion_model = distortion_model
        self.rate_hz = rate_hz
        self.resolution = resolution
        self.aabb = aabb
        self.depth_scale = depth_scale

class ImuCalibration:
    def __init__(self, body_T_imu, a_n, a_b, g_n, g_b, rate_hz, imu_integration_sigma, imu_time_shift, n_gravity) -> None:
        self.body_T_imu = body_T_imu
        self.a_n = a_n
        self.g_n = g_n
        self.a_b = a_b
        self.g_b = g_b
        self.rate_hz = rate_hz
        self.imu_integration_sigma = imu_integration_sigma
        self.imu_time_shift = imu_time_shift
        self.n_gravity = n_gravity
        pass

class ViconCalibration:
    def __init__(self, body_T_vicon) -> None:
        self.body_T_vicon = body_T_vicon
