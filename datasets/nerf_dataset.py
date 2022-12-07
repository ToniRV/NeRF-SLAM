#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import json
import numpy as np
from tqdm import tqdm

from datasets.dataset import * 
from utils.utils import *

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

    def _get_data_packet(self, k0, k1=None):
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
            depth_path = self.depth_paths[i] # index with i, bcs we sorted image_paths to have increasing timestamps.
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


    # Up to you how you index the dataset depending on your training procedure
    def _build_dataset_index(self):
        # Go through the stream and bundle as you wish
        self.data_packets = [data_packet for data_packet in self.stream()]

    def stream(self):
        for k in range(self.__len__()):
            yield self._get_data_packet(k)
