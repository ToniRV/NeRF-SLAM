
import glob
import os
import json
import numpy as np

import cv2
from tqdm import tqdm

from icecream import ic
from datasets.dataset import * 

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

        self.resize_images = False
        if self.calib.resolution.total() > 640*640:
            self.resize_images = True
            # TODO(Toni): keep aspect ratio, and resize max res to 640
            self.output_image_size = [341, 640] # h, w 

        if self.resize_images:
            h0, w0  = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

        subset_poses = []        

        if self.final_k == -1:
            self.final_k = len(self.poses) - 1

        # Parse images and tfs
        for i, (image_path, depth_path) in enumerate(tqdm(zip(self.image_paths, self.depth_paths))):
                
            if ((i-self.initial_k) % self.img_stride) != 0 or i < self.initial_k or i > self.final_k:
                continue

            # Parse rgb/depth images
            image = cv2.imread(image_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            #depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[..., None] # H, W, C=1            

            if self.resize_images:
                w1, h1 = self.w1, self.h1
                image = cv2.resize(image, (w1, h1))
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there

                depth = cv2.resize(depth, (w1, h1))
                depth = depth[:, :, np.newaxis]

                if self.viz:
                    cv2.imshow(f"Img Resized", image)
                    cv2.imshow(f"Depth Resized", depth)
                    cv2.waitKey(1)
            
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
            subset_poses    += [self.poses[i]]

            # Early break if we've exceeded the buffer max
            if len(self.images) == self.args.buffer:
                break

        self.poses = subset_poses

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
