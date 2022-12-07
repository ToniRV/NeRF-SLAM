import os
import cv2
import pandas as pd

from datasets.dataset import * 

# TODO: Didn't finish to implement
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

    