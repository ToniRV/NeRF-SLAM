
import pyrealsense2 as rs
import numpy as np
import cv2
import os

import tqdm

from datasets.dataset import *
from icecream import ic

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