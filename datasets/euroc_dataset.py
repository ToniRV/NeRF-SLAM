#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from icecream import ic

import open3d as o3d
import cv2

from utils.utils import *
from datasets.dataset import * 

class EurocDataset(Dataset):
    yaml = YAML()

    def __init__(self, args, device) -> None:
        super().__init__("Euroc", args, device)

        self.show_gt_pcl = True

        self.parse_metadata(self.dataset_dir)
        self.tqdm = tqdm(total=self.__len__()) # Call after parsing metadata

        # This is in case you want to parse all the dataset first.
        build_dataset_index = False
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

    def parse_metadata(self, dataset_dir):
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
        self.cam_calib : CameraCalibration = self._get_cam_calib(cam0_calib_file)
        self.original_cam_calib : CameraCalibration = self._get_cam_calib(cam0_calib_file)

        self.resize_images = True
        self.output_image_size = [384, 512]
        if self.resize_images:
            h0, w0  = self.cam_calib.resolution.height, self.cam_calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.cam_calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.cam_calib.resolution = Resolution(self.w1, self.h1)

        ## Get Image Lists
        img0_file_list = sorted(os.listdir(self.cam0_data_dir))

        # Build dicts
        # Clean up the file list
        img0_file_dict={}
        for i, img0_file_name in enumerate(img0_file_list):
            t_cam0 = int(os.path.splitext(img0_file_name)[0])
            img0_file_dict[t_cam0] = img0_file_name

        if self.final_k > len(img0_file_list):
            print(f"WARNING: final_k is larger than the number of images in the dataset. Setting final_k to {self.final_k}")
            self.final_k = len(img0_file_list)

        self.img0_file_list = img0_file_list[self.initial_k:self.final_k:self.img_stride]

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
            self.gt_pointcloud = PointCloudTransmissionFormat(o3d.io.read_point_cloud(os.path.join(mav_dir, 'pointcloud0/data_intensity_crop.ply')))
            print("Loaded ply point cloud")

    def _get_data_packet(self, k0, k1=None):
        if k1 is None: 
            k1 = k0 + 1
        else:
            assert(k1 >= k0)

        timestamps = []
        poses      = []
        images     = []
        depths     = []
        calibs     = []

        W, H = self.cam_calib.resolution.width, self.cam_calib.resolution.height 

        for k in np.arange(k0, k1):
            img0_file_name = self.img0_file_list[k]

            # Get timestamp
            t_cam0 = int(os.path.splitext(img0_file_name)[0])

            # Get pose
            t_cam0_near = self.gt_df.index.get_indexer([t_cam0], method="nearest")[0]
            w2c = get_pose_from_df(self.gt_df.iloc[t_cam0_near])

            # Get image
            image = cv2.imread(os.path.join(self.cam0_data_dir, img0_file_name))

            if self.viz:
                cv2.imshow("Img", image)
                cv2.waitKey(1)

            # Undistort img
            image = cv2.undistort(image,
                                  self.original_cam_calib.camera_model.matrix(),
                                  self.original_cam_calib.distortion_model.get_distortion_as_vector())

            if self.viz:
                cv2.imshow("Img Undistort", image)
                cv2.waitKey(1)

            # Resize images
            if self.resize_images:
                image = cv2.resize(image, (self.w1, self.h1))

                if self.viz:
                    cv2.imshow(f"Img Undistort(Rectify) + Resized", image)
                    cv2.waitKey(1)

            # TODO: ideally allocate these in the visual frontend buffered
            # 1st dimension is to add batch dimension (equivalent to unsqueeze(0))
            # 2nd dimension is stereo camera index
            # 3rd dimension is channel
            # 4th, 5th dimensions are height, width
            # This copies the data because of advanced indexing
            # image.shape == (c, 480, 752, 3)
            # img_norm.shape == (1, c, 3, 480, 752)
            assert(H == image.shape[0])
            assert(W == image.shape[1])
            assert(3 == image.shape[2] or 4 == image.shape[2])
            assert(np.uint8 == image.dtype)

            timestamps += [k]
            poses      += [w2c]
            images     += [image]
            depths     += [None]
            calibs     += [self.cam_calib]

        return {"k":      np.arange(k0, k1),
                "t_cams": np.array(timestamps),
                "images": np.array(images),
                "depths": np.array(depths),
                "poses":  np.array(poses),
                "calibs": np.array(calibs),
                "is_last_frame": (k0 >= self.__len__() - 1),
                }

    def __len__(self):
        return len(self.img0_file_list)

    def __getitem__(self, k):
        self.tqdm.update(1)
        return self._get_data_packet(k) if self.data_packets is None else self.data_packets[k]

    # Up to you how you index the dataset depending on your training procedure
    def _build_dataset_index(self):
        # Go through the stream and bundle as you wish
        # Here we do the simplest scenario, send imu data between frames, and the next frame as a packet
        self.data_packets = [data_packet for data_packet in self.stream()]

    def stream(self):
        for k in range(self.__len__()):
            yield self._get_data_packet(k)

    def to_nerf_format(self):
        import math
        OUT_PATH = "transforms.json"
        AABB_SCALE = 4
        out = {
            "fl_x": self.cam_calib.camera_model.fx,
            "fl_y": self.cam_calib.camera_model.fy,
            "k1": self.cam_calib.distortion_model.k1,
            "k2": self.cam_calib.distortion_model.k2,
            "p1": self.cam_calib.distortion_model.p1,
            "p2": self.cam_calib.distortion_model.p2,
            "cx": self.cam_calib.camera_model.cx,
            "cy": self.cam_calib.camera_model.cy,
            "w": self.cam_calib.resolution.width,
            "h": self.cam_calib.resolution.height,
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
            world_T_cam0 =  world_T_body @ self.cam_calib.body_T_cam

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

