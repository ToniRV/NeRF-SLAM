#!/usr/bin/env python3
from abc import abstractclassmethod
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm

import os
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from icecream import ic

import open3d as o3d

from torch.utils.data.dataset import Dataset

import cv2

from utils.utils import *

class _PointCloudTransmissionFormat:
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

class EurocDataset(Dataset):
    yaml = YAML()

    def __init__(self, args, device) -> None:
        super().__init__("Euroc", args, device)

        self.t0 = None # timestamp for first frame in data_packet

        self.show_gt_pcl = True

        build_dataset_index = False

        self._parse_dataset(self.dataset_dir)
        if build_dataset_index:
            print('Building dataset index')
            self._build_dataset_index()
            print('Done building dataset index')

    def undistort_rectify(self, img0, img1):
        K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3)
        K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3)

        d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
        d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)

        R_l = np.array([
            0.999966347530033, -0.001422739138722922, 0.008079580483432283,
            0.001365741834644127, 0.9999741760894847, 0.007055629199258132,
            -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
        ]).reshape(3,3)
        R_r = np.array([
            0.9999633526194376, -0.003625811871560086, 0.007755443660172947,
            0.003680398547259526, 0.9999684752771629, -0.007035845251224894,
            -0.007729688520722713, 0.007064130529506649, 0.999945173484644
        ]).reshape(3,3)

        P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
        P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)

        map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

        return cv2.remap(img0, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),\
               cv2.remap(img1, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    def _get_cam_calib(self, cam_calib_file):
        cam_calib = self.yaml.load(open(cam_calib_file, 'r'))
        T_BS = np.array(cam_calib['T_BS']['data']).reshape(4,4)
        body_T_cam0 = T_BS#SE3(th.as_tensor(pose_matrix_to_t_and_quat(T_BS), device=self.device))
        rate_hz = cam_calib['rate_hz']
        width, height = cam_calib['resolution'][:]
        fx, fy, cx, cy = cam_calib['intrinsics'][:]
        #k1, k2, p2, p1 = cam_calib['distortion_coefficients'][:] # According to the Euroc paper, these are k1, k2, p2(!), p1(!) !!
        k1, k2, p1, p2 = cam_calib['distortion_coefficients'][:] # According to the Euroc paper, these are k1, k2, p2(!), p1(!) !!
        assert(cam_calib['camera_model'] == 'pinhole')
        assert(cam_calib['distortion_model'] == 'radial-tangential')
        resolution = Resolution(width, height)
        pinhole0 = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)
        aabb = np.array([[0,0,0],[1,1,1]])
        depth_scale = 1.0
        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)

    def _get_imu_calib(self, imu_calib_file):
        imu_calib = self.yaml.load(open(imu_calib_file, 'r'))
        T_BS = np.array(imu_calib['T_BS']['data']).reshape(4,4)
        body_T_imu0 = T_BS#SE3(th.as_tensor(pose_matrix_to_t_and_quat(T_BS), device=self.device).unsqueeze(0))
        # ic(SE3.IdentityLike(body_T_imu0).data)
        # assert(body_T_imu0 == SE3.IdentityLike(body_T_imu0))
        imu_rate = imu_calib['rate_hz']
        g_n = imu_calib['gyroscope_noise_density'] # [ rad / s / sqrt(Hz) ] ( gyro "white noise" )
        g_b = imu_calib['gyroscope_random_walk'] # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
        a_n = imu_calib['accelerometer_noise_density'] # [ m / s^2 / sqrt(Hz) ] ( accel "white noise" )
        a_b = imu_calib['accelerometer_random_walk'] # [ m / s^3 / sqrt(Hz) ]. ( accel bias diffusion )
        imu_integration_sigma = None
        if 'imu_integration_sigma' in imu_calib:
            imu_integration_sigma = imu_calib['imu_integration_sigma']
        else:
            print("ERROR: no 'imu_integration_sigma' in calib.")
        imu_time_shift = None
        if 'imu_time_shift' in imu_calib:
            imu_time_shift = imu_calib['imu_time_shift']
        else:
            print("ERROR: no 'imu_time_shift' in calib.")
        n_gravity = None
        if 'n_gravity' in imu_calib:
            n_gravity = np.array(imu_calib['n_gravity'])
        else:
            print("ERROR: no 'n_gravity' in calib.")
        return ImuCalibration(body_T_imu0, a_n, a_b, g_n, g_b, imu_rate, imu_integration_sigma, imu_time_shift, n_gravity)

    def _get_vicon_calib(self, vicon_calib_file):
        vicon_calib = self.yaml.load(vicon_calib_file)
        T_BS = np.array(vicon_calib['T_BS']['data']).reshape(4,4)
        body_T_vicon0 = T_BS#SE3(th.as_tensor(pose_matrix_to_t_and_quat(T_BS), device=self.device))
        return ViconCalibration(body_T_vicon0)

    def _parse_dataset(self, dataset_dir):
        mav_dir = os.path.join(dataset_dir, 'mav0')

        # Cam0
        cam0_dir = os.path.join(mav_dir, 'cam0')
        self.cam0_data_dir = os.path.join(cam0_dir, 'data')
        cam0_calib_file = os.path.join(cam0_dir, 'sensor.yaml')
        #cam0_data_csv = os.path.join(cam0_dir, 'data.csv')

        # Cam1
        cam1_dir = os.path.join(mav_dir, 'cam1')
        self.cam1_data_dir = os.path.join(cam1_dir, 'data')
        cam1_calib_file = os.path.join(cam1_dir, 'sensor.yaml')
        #cam1_data_csv = os.path.join(cam1_dir, 'data.csv')

        ## Get Cam Calib
        self.cam0_calib : CameraCalibration = self._get_cam_calib(cam0_calib_file)
        self.cam_calibs = [self.cam0_calib]
        if self.stereo:
            self.cam1_calib : CameraCalibration = self._get_cam_calib(cam1_calib_file)
            self.cam_calibs += [self.cam1_calib]

        self.resize_images = True
        self.output_image_size = [384, 512]
        if self.resize_images:
            h0, w0  = self.cam0_calib.resolution.height, self.cam0_calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.cam0_calib_resized : CameraCalibration = self._get_cam_calib(cam0_calib_file)
            self.cam0_calib_resized.camera_model.scale_intrinsics(w1 / w0, h1 / h0)
            self.cam_calibs_resized = [self.cam0_calib_resized]
            if self.stereo:
                self.cam1_calib_resized : CameraCalibration = self._get_cam_calib(cam1_calib_file)
                self.cam1_calib_resized.camera_model.scale_intrinsics(w1 / w0, h1 / h0)
                self.cam_calibs_resized += [self.cam1_calib_resized]


        ## Get Image Lists
        img0_file_list = sorted(os.listdir(self.cam0_data_dir))
        if self.stereo:
            img1_file_list = sorted(os.listdir(self.cam1_data_dir))        

        # Build dicts
        # Clean up the file list
        img0_file_dict={}
        for i, img0_file_name in enumerate(img0_file_list):
            t_cam0 = int(os.path.splitext(img0_file_name)[0])
            img0_file_dict[t_cam0] = img0_file_name

        img1_file_dict={}
        if self.stereo:
            for i, img1_file_name in enumerate(img1_file_list):
                t_cam1 = int(os.path.splitext(img1_file_name)[0])
                img1_file_dict[t_cam1] = img1_file_name

        if self.final_k > len(img0_file_list):
            print(f"WARNING: final_k is larger than the number of images in the dataset. Setting final_k to {self.final_k}")
            self.final_k = len(img0_file_list)

        if self.stereo:
            if self.final_k > len(img1_file_list):
                print(f"WARNING: final_k is larger than the number of images in the dataset. Setting final_k to {self.final_k}")
                self.final_k = len(img1_file_list)

        self.img0_file_list = img0_file_list[self.initial_k:self.final_k:self.img_stride]
        if self.stereo:
            self.img1_file_list = img1_file_list[self.initial_k:self.final_k:self.img_stride]
            assert(len(self.img0_file_list) == len(self.img1_file_list))

        # Imu
        imu0_dir = os.path.join(mav_dir, 'imu0')
        imu0_calib_file = os.path.join(imu0_dir, 'sensor.yaml')
        imu0_data_csv = os.path.join(imu0_dir, 'data.csv')

        ## Get IMU Calib
        self.imu_calib = self._get_imu_calib(imu0_calib_file)

        ## Get IMU data
        self.imu_df = pd.read_csv(os.path.join(imu0_dir, 'data.csv'))
        self.imu_df.set_index('#timestamp [ns]', drop=False, inplace=True)

        # Vicon
        vicon0_dir = os.path.join(mav_dir, 'vicon0')
        vicon0_data_csv = os.path.join(vicon0_dir, 'data.csv')
        #vicon0_calib_file = os.path.join(vicon0_dir, 'sensor.yaml')

        ## Get Vicon data
        self.vicon_df = pd.read_csv(vicon0_data_csv)
        self.vicon_df.set_index('#timestamp [ns]',drop=False, inplace=True)

        ## Get ground-truth pose, velocities, biases.
        gt_dir = os.path.join(mav_dir, 'state_groundtruth_estimate0')
        self.gt_df = pd.read_csv(os.path.join(gt_dir, 'data.csv'))
        self.gt_df.set_index('#timestamp', drop=False, inplace=True)
        renamed_cols = {'#timestamp': 'timestamp',
                        ' p_RS_R_x [m]': 'tx',
                        ' p_RS_R_y [m]': 'ty',
                        ' p_RS_R_z [m]': 'tz',
                        ' q_RS_x []': 'qx',
                        ' q_RS_y []': 'qy',
                        ' q_RS_z []': 'qz',
                        ' q_RS_w []': 'qw'}
        self.gt_df.rename(columns=renamed_cols, inplace=True)

        ## Get ground-truth mesh
        self.gt_pointcloud = None
        if self.show_gt_pcl:
            print("Loading ply point cloud")
            self.gt_pointcloud = _PointCloudTransmissionFormat(o3d.io.read_point_cloud(os.path.join(mav_dir, 'pointcloud0/data_intensity_crop.ply')))
            print("Loaded ply point cloud")

    def _get_data_packet(self, k, img0_file_name, img1_file_name=None):
        # The img_filename has the timestamp of the image! At least for Euroc!
        t_cam0 = int(os.path.splitext(img0_file_name)[0])
        if self.stereo:
            t_cam1 = int(os.path.splitext(img1_file_name)[0])
            # delta_t_btw_imu_img = 0 # t_imu = t_img + delta_t TODO
            assert(t_cam0 == t_cam1)
        # Send to GPU?
        t_cams = [t_cam0]
        if self.stereo:
            t_cams += [t_cam1]

        # TODO: I'm not sure we should specify/use th.float64 here, maybe let it figure it out.
        # Group IMU data into packets btw frames
        imu_t0_t1 = None
        gt_t0_t1 = vicon_t0_t1 = pd.DataFrame()
        t1 = t_cam0
        t1_near = self.gt_df.index.get_indexer([t1], method="nearest")[0]
        if self.t0 is not None:
            # +1 to include t1_near, we are duplicating the last row though...
            imu_t0_t1   = self.imu_df.iloc[self.t0:t1_near+1].to_numpy(dtype=np.float64)
            gt_t0_t1    = self.gt_df.iloc[self.t0:t1_near+1]
            vicon_t0_t1 = self.vicon_df.iloc[self.t0:t1_near+1]
        else:
            imu_t0_t1   = self.imu_df.iloc[t1_near].to_numpy(dtype=np.float64)
            gt_t0_t1    = self.gt_df.iloc[t1_near]
            vicon_t0_t1 = self.vicon_df.iloc[t1_near]
        self.t0 = t1_near

        # TODO: ideally load directly to cuda, and do everything in GPU
        # TODO: The channels are redundant, since we are dealing with grey images!!
        # Read images
        images = [cv2.imread(os.path.join(self.cam0_data_dir, img0_file_name))]
        if self.stereo:
            images += [cv2.imread(os.path.join(self.cam1_data_dir, img1_file_name))]

        if self.viz:
            for i, img in enumerate(images):
                cv2.imshow(f"Img{i} Euroc", img)

        # TODO: the undistortion/rectification removes a lot of the image...
        # It would be better to just run on the distorted images, and then
        # undisort the flow.
        # Undistort/Rectify images
        # TODO use the values in self.cam_calibs object instead
        # for i in range(len(images)):
        #     images[i] = cv2.undistort(images[i],
        #                               self.cam_calibs[i].camera_model.get_intrinsics_as_matrix(),
        #                               self.cam_calibs[i].distortion_model.get_distortion_as_vector())
        images = [cv2.undistort(img,
                                self.cam_calibs[i].camera_model.matrix(),
                                self.cam_calibs[i].distortion_model.get_distortion_as_vector())
                  for i, img in enumerate(images)]

        #images[0] = cv2.undistort(images[0],
        #                            self.cam_calibs[0].camera_model.get_intrinsics_as_matrix(),
        #                            self.cam_calibs[0].distortion_model.get_distortion_as_vector())
        #img0, img1 = self.undistort_rectify(img0, img1)

        if self.viz:
            for i, img in enumerate(images):
                cv2.imshow(f"Img{i} Undistort(Rectify)", img)

        # Resize images
        if self.resize_images:
            h0, w0, _ = images[0].shape
            output_image_size = [384, 512]
            total_output_pixels = (output_image_size[0] * output_image_size[1])
            h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))

            for i in range(len(images)):
                images[i] = cv2.resize(images[i], (w1, h1))
                images[i] = images[i][:h1-h1%8, :w1-w1%8]

            if self.viz:
                for i, img in enumerate(images):
                    cv2.imshow(f"Img{i} Undistort(Rectify) + Resized", img)

        # TODO: ideally allocate these in the visual frontend buffered
        # First dimension is to add batch dimension (equivalent to unsqueeze(0))
        # Second dimension is stereo camera index
        # Third dimension is channel
        # Forth,Fifth dimensions are height, width
        # This copies the data because of advanced indexing
        # image.shape == (c, 480, 752, 3)
        # img_norm.shape == (1, c, 3, 480, 752)

        if self.viz:
            cv2.waitKey(1)
        images = np.array(images)
        t_cams = np.array(t_cams)
        assert images.shape[0] == t_cams.shape[0]
        return {"k": k,
                "t_cams": t_cams,
                "images": images,
                "cam_calibs": self.cam_calibs if not self.resize_images else self.cam_calibs_resized,
                "imu_t0_t1": imu_t0_t1,
                "imu_calib": self.imu_calib,
                "gt_t0_t1": gt_t0_t1,
                "vicon_t0_t1": vicon_t0_t1,
                "is_last_frame": (k >= self.__len__() - 1),
                }


    # Return all data btw frames, plus the subsequent frame.
    def stream(self):
        if self.stereo:
            for k, img_file_names in enumerate(zip(self.img0_file_list, self.img1_file_list)):
                yield self._get_data_packet(k, img_file_names)
        else:
            for k, img0_file_name in enumerate(self.img0_file_list):
                yield self._get_data_packet(k, img0_file_name)

    # Up to you how you index the dataset depending on your training procedure
    def _build_dataset_index(self):
        # Go through the stream and bundle as you wish
        # Here we do the simplest scenario, send imu data between frames,
        # and the next frame as a packet
        self.data_packets = [data_packet for data_packet in self.stream()]

    def __getitem__(self, index):
        return self.data_packets[index] if self.data_packets is not None else self._get_data_packet(index, self.img0_file_list[index], self.img1_file_list[index] if self.stereo else None)

    def __len__(self):
        return len(self.data_packets) if self.data_packets is not None else len(self.img0_file_list)

    def to_nerf_format(self):
        import math
        OUT_PATH = "transforms.json"
        AABB_SCALE = 4
        out = {
            "fl_x": self.cam0_calib.camera_model.fx,
            "fl_y": self.cam0_calib.camera_model.fy,
            "k1": self.cam0_calib.distortion_model.k1,
            "k2": self.cam0_calib.distortion_model.k2,
            "p1": self.cam0_calib.distortion_model.p1,
            "p2": self.cam0_calib.distortion_model.p2,
            "cx": self.cam0_calib.camera_model.cx,
            "cy": self.cam0_calib.camera_model.cy,
            "w": self.cam0_calib.resolution.width,
            "h": self.cam0_calib.resolution.height,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }
        out["camera_angle_x"] = math.atan(out["w"] / (out["fl_x"] * 2)) * 2
        out["camera_angle_y"] = math.atan(out["h"] / (out["fl_y"] * 2)) * 2

        up = np.zeros(3)
        if self.data_packets is None:
            self._build_dataset_index()
        for data_packet in self.data_packets:
            gt_df = data_packet["gt_t0_t1"]
            if gt_df.empty:
                continue

            nearest_t_cam0 = gt_df.index.get_indexer([data_packet["t_cams"][0]], method="nearest")[0]
            world_T_body = get_pose_from_df(gt_df.iloc[nearest_t_cam0])
            world_T_cam0 =  world_T_body @ self.cam0_calib.body_T_cam

            # Img name
            cam0_file_name = self.img0_file_list[data_packet["k"]]
            img0_file_path = os.path.join(self.cam0_data_dir, cam0_file_name)

            #TODO(TONI): Add cam1 data!

            # Sharpness
            b = sharpness(data_packet["images"][0])
            print(cam0_file_name, "sharpness =",b)

            # Transform
            c2w = np.linalg.inv(world_T_cam0)
            c2w = world_T_cam0
            # Convert from opencv convention to nerf convention
            c2w[0:3, 1] *= -1 # flip the y axis
            c2w[0:3, 2] *= -1 # flip the z axis
            #c2w = c2w[[1, 0, 2, 3],:] # swap y and z
            #c2w[2, :] *= -1 # flip whole world upside down

            up += c2w[0:3, 1]

            frame = {"file_path": img0_file_path,
                     "sharpness": b,
                     "transform_matrix": c2w}
            out["frames"].append(frame)

        nframes = len(out["frames"])
        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in tqdm(out["frames"]):
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.01:
                    totp += p * w
                    totw += w
        totp /= totw
        print(totp) # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()
        print(nframes,"frames")
        print(f"writing {OUT_PATH}")

        with open(OUT_PATH, "w") as outfile:
            import json
            json.dump(out, outfile, indent=2)


class TumDataset(Dataset):
    def __init__(self, args, device):
        super().__init__("Tum", args, device)
        self.dataset_dir = args.dataset_dir

        json_file = os.path.join(self.dataset_dir, "associations.txt")
        self.associations = open(associations_file).readlines()
        assert self.associations is not None

        self.resize_images = False

        self.t0 = None

        self.final_k = self.final_k if self.final_k != -1.0 and self.final_k < len(
            self.associations) else len(self.associations)

        self.parse_dataset()

    def parse_dataset(self):
        ## Get Cam Calib
        self.cam0_calib : CameraCalibration = self._get_cam_calib()
        self.cam_calibs = [self.cam0_calib]

        ## Get ground-truth pose
        gt_dir = os.path.join(self.dataset_dir, 'groundtruth.txt')
        self.gt_df = pd.read_csv(gt_dir, sep=" ", comment="#",
                                 names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        self.gt_df.timestamp *= 10000
        self.gt_df.timestamp = self.gt_df['timestamp'].astype('int64')
        self.gt_df.set_index("timestamp", drop=False, inplace=True)

    def get_rgb(self, frame_id):
        return self._get_img(frame_id, 'rgb')

    def get_depth(self, frame_id):
        return self._get_img(frame_id, 'depth')

    def _get_img(self, frame_id, type):
        if frame_id >= self.final_k:
            return None, None

        row = self.associations[frame_id].strip().split()
        if type == 'rgb':
            img_file_name = row[1]
        elif type == 'depth':
            img_file_name = row[3]
        else:
            raise "Unknown img type"

        timestamp = float(row[0])
        img = cv2.imread(os.path.join(self.dataset_dir, img_file_name))

        assert img is not None

        return timestamp, img

    def _get_data_packet(self, k):
        # The img_filename has the timestamp of the image! At least for Euroc!
        t_rgb0, img_0 = self.get_rgb(k)
        t_depth0, depth_0 = self.get_depth(k)
        assert t_rgb0 == t_depth0
        t_cam0 = t_rgb0

        t_cams = [t_cam0]
        images = [img_0]
        depths = [depth_0]

        if self.viz:
            for i, (rgb, depth) in enumerate(zip(images, depths)):
                if rgb is not None:
                    cv2.imshow(f"Img{i}", rgb)
                if depth is not None:
                    cv2.imshow(f"Depth{i}", depth)

        # Resize images
        if self.resize_images:
            h0, w0, _ = images[0].shape
            output_image_size = [384, 512]
            total_output_pixels = (output_image_size[0] * output_image_size[1])
            h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))

            for i in range(len(images)):
                images[i] = cv2.resize(images[i], (w1, h1))
                images[i] = images[i][:h1-h1%8, :w1-w1%8]

                # TODO: can we do this for depths??
                depths[i] = cv2.resize(depths[i], (w1, h1))
                depths[i] = depths[i][:h1-h1%8, :w1-w1%8]

            if self.viz:
                for i, rgb, depth in enumerate(zip(images, depths)):
                    cv2.imshow(f"Img{i} Resized", rgb)
                    cv2.imshow(f"Depth{i} Resized", depth)

        if self.viz:
            cv2.waitKey(1)

        t_cams = np.array(t_cams)
        images = np.array(images)
        assert len(images) == len(t_cams)
        assert len(depths) == len(t_cams)

        # Ground-truth
        #t1_gt_near = self.gt_df['timestamp'].sub(t1).abs().idxmin()
        #t0_gt_near = self.gt_df['timestamp'].sub(self.t0).abs().idxmin()
        t1 = t_cam0
        t1_near = self.gt_df.index.get_indexer([t1], method="nearest")[0]
        if self.t0 is not None:
            gt_t0_t1 = self.gt_df.iloc[self.t0:t1_near+1] # +1 to include t1
        else:
            gt_t0_t1 = self.gt_df.iloc[t1_near]
        self.t0 = t1_near

        return {"k": k, "t_cams": t_cams, "images": images,
                "cam_calibs": self.cam_calibs if not self.resize_images else self.cam_calibs_resized,
                "gt_t0_t1": gt_t0_t1,
                "is_last_frame": (k >= self.__len__() - 1)}

    def _get_cam_calib(self):
        # TODO: remove hardcoded
        body_T_cam0    = np.eye(4)
        rate_hz        = 0.0
        width, height  = 640, 480
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

        resolution = Resolution(width, height)
        pinhole0 = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)

        aabb = np.array([[0,0,0],[1,1,1]])
        depth_scale = 1.0

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)

    # Up to you how you index the dataset depending on your training procedure
    def _build_dataset_index(self):
        # Go through the stream and bundle as you wish
        # Here we do the simplest scenario, send imu data between frames,
        # and the next frame as a packet
        self.data_packets = [data_packet for data_packet in self.stream()]

    def __getitem__(self, index):
        return self.data_packets[index] if self.data_packets is not None else self._get_data_packet(index)

    def __len__(self):
        return len(self.data_packets) if self.data_packets is not None else len(self.associations)

    # Return all data btw frames, plus the subsequent frame.
    def stream(self):
        for k in self.__len__():
            yield self._get_data_packet(k)

    
import json
class NeRFDataset(Dataset):
    def __init__(self, args, device):
        super().__init__("Nerf", args, device)
        self.parse_metadata()
        # self._build_dataset_index() # Loads all the data first, and then streams.
        self.tqdm = tqdm(total=self.__len__()) # Call after parsing metadata

    def get_cam_calib(self):
        w, h   = self.json["w"],    self.json["h"]
        fx, fy = self.json["fl_x"], self.json["fl_y"]
        cx, cy = self.json["cx"],   self.json["cy"]

        body_T_cam0 = np.eye(4,4)
        rate_hz = 10.0
        resolution = Resolution(w, h)
        pinhole0 = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(0, 0, 0, 0)

        aabb = self.json["aabb"]
        depth_scale = self.json["integer_depth_scale"] if "integer_depth_scale" in self.json else 1.0

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)

    def parse_metadata(self):
        with open(os.path.join(self.dataset_dir, "transforms.json"), 'r') as f:
            self.json = json.load(f)

        self.calib = self.get_cam_calib()

        self.resize_images = False
        if self.calib.resolution.total() > 640*640:
            self.resize_images = True
            # TODO(Toni): keep aspect ratio, and resize max res to 640
            self.output_image_size = [341, 640] # h, w 

        self.image_paths = []
        self.depth_paths = []
        self.w2c = []

        if self.resize_images:
            h0, w0  = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

        frames = self.json["frames"]
        frames = frames[self.initial_k:self.final_k:self.img_stride]
        print(f'Loading {len(frames)} images.')
        for i, frame in enumerate(frames):
            # Convert from nerf to ngp
            # TODO: convert poses to our format
            c2w = np.array(frame['transform_matrix'])
            c2w = nerf_matrix_to_ngp(c2w) # THIS multiplies by scale = 1 and offset = 0.5
            # TODO(TONI): prone to numerical errors, do se(3) inverse instead
            w2c = np.linalg.inv(c2w)

            # Get rgb/depth images path
            if frame['file_path'].endswith(".png") or frame['file_path'].endswith(".jpg"):
                image_path = os.path.join(self.dataset_dir, f"{frame['file_path']}")
            else:
                image_path = os.path.join(self.dataset_dir, f"{frame['file_path']}.png")
            depth_path = None
            if 'depth_path' in frame:
                depth_path = os.path.join(self.dataset_dir, f"{frame['depth_path']}")

            self.image_paths.append([i, image_path])
            self.depth_paths += [depth_path]
            self.w2c += [w2c]

        # Sort paths chronologically
        if os.path.splitext(os.path.basename(self.image_paths[0][1]))[0].isdigit():
            # name is "000000.jpg" for Cube-Diorama
            sorted(self.image_paths, key=lambda path: int(os.path.splitext(os.path.basename(path[1]))[0]))
        else:
            # name is "frame000000.jpg" for Replica
            sorted(self.image_paths, key=lambda path: int(os.path.splitext(os.path.basename(path[1]))[0][5:]))

        # Store the first pose, used as prior and initial state in SLAM.
        self.args.world_T_imu_t0 = self.w2c[0]

    def read_data(self, k0, k1=None):
        if k1 is None: 
            k1 = k0 + 1
        else:
            assert(k1 >= k0)

        timestamps = []
        poses      = []
        images     = []
        depths     = []
        calibs = []

        W, H = self.calib.resolution.width, self.calib.resolution.height 

        # Parse images and tfs
        for k in np.arange(k0, k1):
            i, image_path = self.image_paths[k]
            depth_path = self.depth_paths[i]
            w2c = self.w2c[i]

            # Parse rgb/depth images
            image = cv2.imread(image_path) # H, W, C=4
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there
            if depth_path:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[..., None] # H, W, C=1
            else:
                depth = (-1 * np.ones_like(image[:, :, 0])).astype(np.uint16) # invalid depth

            if self.resize_images:
                w1, h1 = self.w1, self.h1
                image = cv2.resize(image, (w1, h1))
                depth = cv2.resize(depth, (w1, h1))
                depth = depth[:, :, np.newaxis]

                if self.viz:
                    cv2.imshow(f"Img Resized", image)
                    cv2.imshow(f"Depth Resized", depth)
                    cv2.waitKey(1)

            assert(H == image.shape[0])
            assert(W == image.shape[1])
            assert(3 == image.shape[2] or 4 == image.shape[2])
            assert(np.uint8 == image.dtype)
            assert(H == depth.shape[0])
            assert(W == depth.shape[1])
            assert(1 == depth.shape[2])
            assert(np.uint16 == depth.dtype)

            depth = depth.astype(np.int32) # converting to int32, because torch does not support uint16, and I don't want to lose precision

            timestamps += [i]
            poses      += [w2c]
            images     += [image]
            depths     += [depth]
            calibs     += [self.calib]

        return {"k":      np.arange(k0,k1),
                "t_cams": np.array(timestamps),
                "poses":  np.array(poses),
                "images": np.array(images),
                "depths": np.array(depths),
                "calibs": np.array(calibs),
                "is_last_frame": (i >= self.__len__() - 1),
                }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, k):
        self.tqdm.update(1)
        return self._get_data_packet(k) if self.data_packets is None else self.data_packets[k]

    def _get_data_packet(self, k0, k1=None):
        return self.read_data(k0, k1)

    # Up to you how you index the dataset depending on your training procedure
    def _build_dataset_index(self):
        # Go through the stream and bundle as you wish
        self.data_packets = [data_packet for data_packet in self.stream()]

    def stream(self):
        for k in range(self.__len__()):
            yield self._get_data_packet(k)

import glob
class ReplicaDataset(Dataset):
    def __init__(self, args, device):
        super().__init__("Replica", args, device)
        self.dataset_dir = args.dataset_dir
        self.parse_dataset()
        self._build_dataset_index()

    def load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(len(self.image_paths)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            w2c = np.linalg.inv(c2w)
            poses.append(w2c)
        return poses

    def _get_cam_calib(self, path):
        with open(os.path.join(self.dataset_dir, "../cam_params.json"), 'r') as f:
            self.json = json.load(f)

        camera = self.json["camera"]
        w, h = camera['w'], camera['h']
        fx, fy, cx, cy= camera['fx'], camera['fy'], camera['cx'], camera['cy']

        k1, k2, p1, p2 = 0, 0, 0, 0
        body_T_cam0 = np.eye(4)
        rate_hz = 0

        resolution  = Resolution(w, h)
        pinhole0    = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)

        aabb = np.array([[-2, -2, -2], [2, 2, 2]]) # Computed automatically in to_nerf()
        depth_scale = 1.0 / camera["scale"] # Since we multiply as gt_depth *= depth_scale, we need to invert camera["scale"]

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)

    def parse_dataset(self):
        self.timestamps = []
        self.poses      = []
        self.images     = []
        self.depths     = []
        self.calibs     = []

        self.image_paths = sorted(glob.glob(f'{self.dataset_dir}/results/frame*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.dataset_dir}/results/depth*.png'))
        self.poses       = self.load_poses(f'{self.dataset_dir}/traj.txt')
        self.calib       = self._get_cam_calib(f'{self.dataset_dir}/../cam_params.json')

        N = self.args.buffer
        H, W = self.calib.resolution.height, self.calib.resolution.width

        # Parse images and tfs
        for i, (image_path, depth_path) in enumerate(tqdm(zip(self.image_paths, self.depth_paths))):
            if i >= N:
                break

            # Parse rgb/depth images
            image = cv2.imread(image_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # this is for NERF
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[..., None] # H, W, C=1

            H, W, _  = depth.shape
            assert(H == image.shape[0])
            assert(W == image.shape[1])
            assert(3 == image.shape[2] or 4 == image.shape[2])
            assert(np.uint8 == image.dtype)
            assert(H == depth.shape[0])
            assert(W == depth.shape[1])
            assert(1 == depth.shape[2])
            assert(np.uint16 == depth.dtype)

            depth = depth.astype(np.int32) # converting to int32, because torch does not support uint16, and I don't want to lose precision

            self.timestamps += [i]
            self.images     += [image]
            self.depths     += [depth]
            self.calibs     += [self.calib]

        self.poses = self.poses[:N]

        self.timestamps = np.array(self.timestamps)
        self.poses      = np.array(self.poses)
        self.images     = np.array(self.images)
        self.depths     = np.array(self.depths)
        self.calibs     = np.array(self.calibs)

        N = len(self.timestamps)
        assert(N == self.poses.shape[0])
        assert(N == self.images.shape[0])
        assert(N == self.depths.shape[0])
        assert(N == self.calibs.shape[0])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, k):
        return self.data_packets[k] if self.data_packets is not None else self._get_data_packet(k)

    def _get_data_packet(self, k0, k1=None):
        if k1 is None: 
            k1 = k0 + 1
        else:
            assert(k1 >= k0)
        return {"k":      np.arange(k0,k1),
                "t_cams": self.timestamps[k0:k1],
                "poses":  self.poses[k0:k1],
                "images": self.images[k0:k1],
                "depths": self.depths[k0:k1],
                "calibs": self.calibs[k0:k1],
                "is_last_frame": (k0 >= self.__len__() - 1),
                }

    # Up to you how you index the dataset depending on your training procedure
    def _build_dataset_index(self):
        # Go through the stream and bundle as you wish
        self.data_packets = [data_packet for data_packet in self.stream()]

    def stream(self):
        for k in range(self.__len__()):
            yield self._get_data_packet(k)

    def to_nerf_format(self):
        print("Exporting Replica dataset to Nerf")
        OUT_PATH = "transforms.json"
        AABB_SCALE = 4
        out = {
            "fl_x": self.calib.camera_model.fx,
            "fl_y": self.calib.camera_model.fy,
            "k1": self.calib.distortion_model.k1,
            "k2": self.calib.distortion_model.k2,
            "p1": self.calib.distortion_model.p1,
            "p2": self.calib.distortion_model.p2,
            "cx": self.calib.camera_model.cx,
            "cy": self.calib.camera_model.cy,
            "w": self.calib.resolution.width,
            "h": self.calib.resolution.height,
            # TODO(Toni): calculate this automatically. Box that fits all cameras +2m
            "aabb": self.calib.aabb,
            "aabb_scale": AABB_SCALE,
            "integer_depth_scale": self.calib.depth_scale,
            "frames": [],
        }

        poses_t = []
        if self.data_packets is None:
            self._build_dataset_index()
        for data_packet in self.data_packets:
            # Image
            ic(data_packet["k"])
            color_path = self.image_paths[data_packet["k"][0]]
            depth_path = self.depth_paths[data_packet["k"][0]]

            relative_color_path = os.path.join("results", os.path.basename(color_path))
            relative_depth_path = os.path.join("results", os.path.basename(depth_path))

            # Transform
            w2c = data_packet["poses"][0]
            c2w = np.linalg.inv(w2c)

            # Convert from opencv convention to nerf convention
            c2w[0:3, 1] *= -1  # flip the y axis
            c2w[0:3, 2] *= -1  # flip the z axis

            poses_t += [w2c[:3,3].flatten()]

            frame = {"file_path": relative_color_path,  # "sharpness": b,
                     "depth_path": relative_depth_path,
                     "transform_matrix": c2w.tolist()}
            out["frames"].append(frame)

        poses_t = np.array(poses_t)
        delta_t = 1.0 # 1 meter extra to allow for the depth of the camera
        t_max = np.amax(poses_t, 0).flatten()
        t_min = np.amin(poses_t, 0).flatten()
        out["aabb"] = np.array([t_min-delta_t, t_max+delta_t]).tolist()

        # Save the path to the ground-truth mesh as well
        out["gt_mesh"] = os.path.join("..", os.path.basename(self.dataset_dir)+"_mesh.ply")
        ic(out["gt_mesh"])

        with open(OUT_PATH, "w") as outfile:
            import json
            json.dump(out, outfile, indent=2)

import pyrealsense2 as rs
import numpy as np
import cv2
class RealSenseDataset(Dataset):

    def __init__(self, args, device):
        super().__init__("RealSense", args, device)
        self.parse_metadata()

    def parse_metadata(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)

        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break

        if not found_rgb:
            raise NotImplementedError("No RGB camera found")

        if device_product_line == 'L500':
            raise NotImplementedError

        self.rate_hz = 30
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.rate_hz)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.rate_hz)

        # Set timestamp
        self.timestamp = 0

        # Start streaming
        ic("Start streaming")
        cfg = self.pipeline.start(config)

        # Set profile
        depth_cam, rgb_cam = cfg.get_device().query_sensors()
        rgb_cam.set_option(rs.option.enable_auto_exposure, False)
        rgb_cam.set_option(rs.option.exposure, 238)
        rgb_cam.set_option(rs.option.enable_auto_white_balance, False)
        rgb_cam.set_option(rs.option.white_balance, 3700)
        rgb_cam.set_option(rs.option.gain, 0)
        depth_cam.set_option(rs.option.enable_auto_exposure, False)
        depth_cam.set_option(rs.option.exposure, 438)
        depth_cam.set_option(rs.option.gain, 0)
        # color_sensor.set_option(rs.option.backlight_compensation, 0) # Disable backlight compensation

        # Set calib
        profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for color stream
        intrinsics = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        self.calib = self._get_cam_calib(intrinsics)

        self.resize_images = True
        if self.resize_images:
            self.output_image_size = [315, 420] # h, w 
            h0, w0  = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

    def _get_cam_calib(self, intrinsics):
        """ intrinsics: 
            model	Distortion model of the image
            coeffs	Distortion coefficients
            fx	    Focal length of the image plane, as a multiple of pixel width
            fy	    Focal length of the image plane, as a multiple of pixel height
            ppx	    Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
            ppy	    Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
            height	Height of the image in pixels
            width	Width of the image in pixels
        """
        w, h = intrinsics.width, intrinsics.height
        fx, fy, cx, cy= intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

        distortion_coeffs = intrinsics.coeffs
        distortion_model  = intrinsics.model
        k1, k2, p1, p2 = 0, 0, 0, 0
        body_T_cam0 = np.eye(4)
        rate_hz = self.rate_hz

        resolution  = Resolution(w, h)
        pinhole0    = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)

        aabb        = (2*np.array([[-2, -2, -2], [2, 2, 2]])).tolist() # Computed automatically in to_nerf()
        depth_scale = 1.0 # TODO # Since we multiply as gt_depth *= depth_scale, we need to invert camera["scale"]

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)


    def stream(self):
        self.viz=True

        timestamps = []
        poses      = []
        images     = []
        depths     = []
        calibs     = []

        got_image = False
        while not got_image:
            # Wait for a coherent pair of frames: depth and color
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as e: 
                print(e)
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            #depth_frame = np.zeros((color_image.shape[0], color_image.shape[1], 1))

            if not depth_frame or not color_frame:
                print("No depth and color frame parsed.")
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())


            if self.resize_images:
                color_image = cv2.resize(color_image, (self.w1, self.h1))
                depth_image = cv2.resize(depth_image, (self.w1, self.h1))

            if self.viz:
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow(f"Color Img", color_image)
                cv2.imshow(f"Depth Img", depth_colormap)
                cv2.waitKey(1)

            self.timestamp += 1
            if self.args.img_stride > 1 and self.timestamp % self.args.img_stride == 0:
                # Software imposed fps to rate_hz/img_stride
                continue

            timestamps += [self.timestamp]
            poses      += [np.eye(4)] # We don't have poses
            images     += [color_image]
            depths     += [depth_image] # We don't use depth
            calibs     += [self.calib]
            got_image  = True

        return {"k":      np.arange(self.timestamp-1,self.timestamp),
                "t_cams": np.array(timestamps),
                "poses":  np.array(poses),
                "images": np.array(images),
                "depths": np.array(depths),
                "calibs": np.array(calibs),
                "is_last_frame": False, #TODO
                }
    
    def shutdown(self):
        # Stop streaming
        self.pipeline.stop()

    def to_nerf_format(self, data_packets):
        print("Exporting RealSense dataset to Nerf")
        OUT_PATH = "transforms.json"
        AABB_SCALE = 4
        out = {
            "fl_x": self.calib.camera_model.fx,
            "fl_y": self.calib.camera_model.fy,
            "k1": self.calib.distortion_model.k1,
            "k2": self.calib.distortion_model.k2,
            "p1": self.calib.distortion_model.p1,
            "p2": self.calib.distortion_model.p2,
            "cx": self.calib.camera_model.cx,
            "cy": self.calib.camera_model.cy,
            "w": self.calib.resolution.width,
            "h": self.calib.resolution.height,
            "aabb": self.calib.aabb,
            "aabb_scale": AABB_SCALE,
            "integer_depth_scale": self.calib.depth_scale,
            "frames": [],
        }

        from PIL import Image

        c2w = np.eye(4).tolist()
        for data_packet in tqdm(data_packets):
            # Image
            ic(data_packet["k"])
            k = data_packet["k"][0]
            i = data_packet["images"][0]
            d = data_packet["depths"][0]

            # Store image paths
            color_path = os.path.join(self.args.dataset_dir, "results", f"frame{k:05}.png")
            depth_path = os.path.join(self.args.dataset_dir, "results", f"depth{k:05}.png")

            # Save image to disk
            color = Image.fromarray(i)
            depth = Image.fromarray(d)
            color.save(color_path)
            depth.save(depth_path)

            # Sharpness
            sharp = sharpness(i)

            # Store relative path
            relative_color_path = os.path.join("results", os.path.basename(color_path))
            relative_depth_path = os.path.join("results", os.path.basename(depth_path))

            frame = {"file_path": relative_color_path, 
                     "sharpness": sharp,
                     "depth_path": relative_depth_path,
                     "transform_matrix": c2w}
            out["frames"].append(frame)

        with open(os.path.join(self.args.dataset_dir, OUT_PATH), "w") as outfile:
            import json
            json.dump(out, outfile, indent=2)
