#!/usr/bin/env python3

import torch
from lietorch import SE3

import numpy as np
import cv2

from icecream import ic

from utils.flow_viz import *

import os
import sys
import glob

# Search for pyngp in the build folder.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(
    os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(
    os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

import pyngp as ngp
import pandas

from utils.utils import *

class NerfFusion:
    def __init__(self, name, args, device) -> None:
        ic(os.environ['CUDA_VISIBLE_DEVICES'])

        self.name = name
        self.args = args
        self.device = device

        self.viz = False

        # self.render_path_i = 0
        # import json
        # with open(os.path.join(args.dataset_dir, "transforms.json"), 'r') as f:
        #     self.json = json.load(f)
        # self.render_path = []
        # self.gt_to_slam_scale = 0.1 # We should be calculating this online.... Sim(3) pose alignment
        # for frame in self.json["frames"]:
        #     c2w = np.array(frame['transform_matrix'])
        #     c2w = nerf_matrix_to_ngp(c2w, scale=self.gt_to_slam_scale, offset=0.0) # THIS multiplies by scale = 1 and offset = 0.5
        #     w2c = np.linalg.inv(c2w)
        #     self.render_path += [w2c]

        self.iters = 1
        self.iters_if_none = 1
        self.total_iters = 0
        self.stop_iters  = 25000
        self.old_training_step = 0

        mode = ngp.TestbedMode.Nerf
        configs_dir = os.path.join(ROOT_DIR, "thirdparty/instant-ngp/configs", "nerf")

        base_network = os.path.join(configs_dir, "base.json")
        network = args.network if args.network else base_network
        if not os.path.isabs(network):
            network = os.path.join(configs_dir, network)

        self.ngp = ngp.Testbed(mode, 0) # NGP can only use device = 0

        n_images = args.buffer
        aabb_scale = 4
        nerf_scale = 1.0 # Not needed unless you call self.ngp.load_training_data() or you render depths!
        offset = np.array([np.inf, np.inf, np.inf]) # Not needed unless you call self.ngp.load_training_data()
        render_aabb = ngp.BoundingBox(np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])) # a no confundir amb self.ngp.aabb/raw_aabb/render_aabb
        self.ngp.create_empty_nerf_dataset(n_images, nerf_scale, offset, aabb_scale, render_aabb)

        self.ngp.nerf.training.n_images_for_training = 0;

        if args.gui:
            # Pick a sensible GUI resolution depending on arguments.
            sw = args.width or 1920
            sh = args.height or 1080
            while sw*sh > 1920*1080*4:
                sw = int(sw / 2)
                sh = int(sh / 2)
            self.ngp.init_window(640, 480, second_window=False)

            # Gui params:
            self.ngp.display_gui = True
            self.ngp.nerf.visualize_cameras = True
            self.ngp.visualize_unit_cube = False

        self.ngp.reload_network_from_file(network)

        # NGP Training params:
        self.ngp.shall_train = True
        self.ngp.dynamic_res = True
        self.ngp.dynamic_res_target_fps = 15
        self.ngp.camera_smoothing = True
        #self.ngp.nerf.training.near_distance = 0.2
        #self.ngp.nerf.training.density_grid_decay = 1.0
        self.ngp.nerf.training.optimize_extrinsics = True
        self.ngp.nerf.training.depth_supervision_lambda = 1.0
        self.ngp.nerf.training.depth_loss_type = ngp.LossType.L2
        self.mask_type = args.mask_type # "ours", "ours_w_thresh" or "raw", "no_depth"

        # Keeps track of frame_ids being reconstructed
        self.ref_frames = {}

        self.mesh_renderer = None

        self.anneal = False
        self.anneal_every_iters = 200
        self.annealing_rate = 0.95

        self.evaluate = args.eval
        self.eval_every_iters = 200
        if self.evaluate:
            self.df = pandas.DataFrame(columns=['Iter', 'Dt','PSNR', 'L1', 'count'])

        # Fit vol once to init gui
        self.fit_volume_once()

    def process_data(self, packet):
        # GROUND_TRUTH Fitting
        self.ngp.nerf.training.optimize_extrinsics = False

        calib = packet["calibs"][0]
        scale, offset = get_scale_and_offset(calib.aabb)
        gt_depth_scale = calib.depth_scale

        packet["poses"]            = scale_offset_poses(np.linalg.inv(packet["poses"]), scale=scale, offset=offset)
        packet["images"]           = (packet["images"].astype(np.float32) / 255.0)
        packet["depths"]           = (packet["depths"].astype(np.float32))
        packet["gt_depths"]        = (packet["depths"].astype(np.float32))
        packet["depth_scale"]      = gt_depth_scale * scale
        packet["depths_cov"]       = np.ones_like(packet["depths"])
        packet["depths_cov_scale"] = 1.0

        self.send_data(packet)
        return False

    def process_slam(self, packet):
        # SLAM_TRUTH Fitting

        # No slam output, just fit for some iters
        if not packet:
            print("Missing fusion input packet from SLAM module...")
            return True

        # Slam output is None, just fit for some iters
        slam_packet = packet[1]
        if slam_packet is None:
            print("Fusion packet from SLAM module is None...")
            return True

        if slam_packet["is_last_frame"]:
            return True

        # Get new data and fit volume
        viz_idx        = slam_packet["viz_idx"]
        cam0_T_world   = slam_packet["cam0_poses"]
        images         = slam_packet["cam0_images"]
        idepths_up     = slam_packet["cam0_idepths_up"]
        depths_cov_up  = slam_packet["cam0_depths_cov_up"]
        calibs         = slam_packet["calibs"]
        gt_depths      = slam_packet["gt_depths"]

        calib = calibs[0]
        scale, offset = get_scale_and_offset(calib.aabb) # if we happen to change aabb, we are screwed...
        gt_depth_scale = calib.depth_scale
        scale = 1.0 # We manually set the scale to 1.0 bcs the automatic get_scale_and_offset sets the scale too small for high-quality recons.
        offset = np.array([0.0, 0.0, 0.0])

        # Mask out depths that have too much uncertainty
        if self.mask_type == "ours":
            pass
        elif self.mask_type == "raw":
            depths_cov_up[...] = 1.0
        elif self.mask_type == "ours_w_thresh":
            masks = (depths_cov_up.sqrt() > depths_cov_up.quantile(0.50))
            idepths_up[masks] = -1.0
        elif self.mask_type == "no_depth":
            idepths_up[...] = -1.0
        else:
            raise NotImplementedError(f"Unknown mask type: {self.mask_type}")

        #TODO: 
        # poses -> matrix
        # images -> [N,H,W,4] float cpu
        # depths -> [N,H,W,1] float cpu up-sampled
        # calibs -> up-sampled
        assert(images.dtype == torch.uint8)
        assert(idepths_up.dtype == torch.float)
        assert(depths_cov_up.dtype == torch.float)

        if self.viz:
            viz_depth_sigma(depths_cov_up.unsqueeze(-1).sqrt(), fix_range=True, bg_img=images, sigma_thresh=20.0, name="Depth Sigma for Fusion")
            cv2.waitKey(1)

        N, _, H, W = images.shape
        alpha_padding = 255 * torch.ones(N, 1, H, W, dtype=images.dtype, device=images.device) # we could avoid this if we didn't remove the alpha channel in the frontend
        images = torch.cat((images, alpha_padding), 1)

        cam0_T_world = SE3(cam0_T_world).matrix().contiguous().cpu().numpy()
        world_T_cam0 = scale_offset_poses(np.linalg.inv(cam0_T_world), scale=scale, offset=offset)
        images = (images.permute(0,2,3,1).float() / 255.0)
        depths = (1.0 / idepths_up[..., None])
        depths_cov = depths_cov_up[..., None]
        gt_depths = gt_depths.permute(0, 2, 3, 1) * gt_depth_scale * scale

        # This is extremely slow.
        # TODO: we could do it in cpp/cuda: send the uint8_t image instead of float, and call srgb_to_linear inside the convert_rgba32 function
        if images.shape[2] == 4:
            images[...,0:3] = srgb_to_linear(images[...,0:3], self.device)
            images[...,0:3] *= images[...,3:4] # Pre-multiply alpha
        else:
            images = srgb_to_linear(images, self.device)

        data_packets = {"k":            viz_idx.cpu().numpy(),
                    "poses":            world_T_cam0,  # needs to be c2w
                    "images":           images.contiguous().cpu().numpy(),
                    "depths":           depths.contiguous().cpu().numpy(),
                    "depth_scale":      scale, # This should be scale, since we scale the poses... # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale,
                    "depths_cov":       depths_cov.contiguous().cpu().numpy(), # do not use up
                    "depths_cov_scale": scale, # , 1.0, #np.mean(depths), #* self.ngp.nerf.training.dataset.scale, 
                    "gt_depths":        gt_depths.contiguous().cpu().numpy(), 
                    "calibs":           calibs,
                }

        # Uncomment in case you want to use ground-truth poses
        # batch["poses"] = np.linalg.inv(gt_poses.cpu().numpy())
        # batch["depths"] = (gt_depths.permute(0,2,3,1).float()).contiguous().cpu().numpy()
        # batch["depth_scale"] = 4.5777065690089265e-05 * self.ngp.nerf.training.dataset.scale
        # gt_pose = np.linalg.inv(gt_poses[0].cpu().numpy()) # c2w gt poses in ngp format

        self.send_data(data_packets)
        return False

    # Main LOOP
    def fuse(self, data_packets):
        fit = False
        if data_packets:  # data_packets is a dict of data_packets
            for name, packet in data_packets.items():
                if name == "data":
                    fit = self.process_data(packet)
                elif name == "slam":
                    fit = self.process_slam(packet)
                    #self.ngp.set_camera_to_training_view(self.ngp.nerf.training.n_images_for_training-1) 
                else:
                    raise NotImplementedError(f"process_{name} not implemented...")
            if fit:
                self.fit_volume()
        else:
            #print("No packet received in fusion module.")
            self.fit_volume()

        # Set the gui to a given pose, and follow the gt pose, but modulate the speed somehow...
        # a) allow to provide a pose (from gt)
        # b) position gui cam there (need the pybind for that) No need! It's camera_matrix!
        # c) allow to speed/slow cam mov (use slerp)
        # TODO: ideally set it slightly ahead!
        #self.render_path_i += 1
        #self.ngp.camera_matrix = self.render_path[self.render_path_i][:3,:]
        return True  # return None if we want to shutdown

    def stop_condition(self):
        return self.total_iters > self.stop_iters if self.evaluate else False

    def send_data(self, batch):
        frame_ids       = batch["k"]
        poses           = batch["poses"]
        images          = batch["images"]
        depths          = batch["depths"]
        depth_scale     = batch["depth_scale"]
        depths_cov      = batch["depths_cov"]
        depth_cov_scale = batch["depths_cov_scale"]
        gt_depths       = batch["gt_depths"]
        calib           = batch["calibs"][0]  # assumes all the same calib

        intrinsics = calib.camera_model.numpy()
        resolution = calib.resolution.numpy()

        focal_length = intrinsics[:2]
        principal_point = intrinsics[2:]

        # TODO: we need to restore the self.ref_frames[frame_id] = [image, gt, etc] for evaluation....
        self.ngp.nerf.training.update_training_images(list(frame_ids),
                                                      list(poses[:, :3, :4]), 
                                                      list(images), 
                                                      list(depths), 
                                                      list(depths_cov), resolution, principal_point, focal_length, depth_scale, depth_cov_scale)

        # On the first frame, set the viewpoint
        if self.ngp.nerf.training.n_images_for_training == 1:
            self.ngp.set_camera_to_training_view(0) 


    def fit_volume(self):
        #print(f"Fitting volume for {self.iters} iters")
        self.fps = 30
        for _ in range(self.iters):
            self.fit_volume_once()
            self.ngp.apply_camera_smoothing(1000.0/self.fps)

    def fit_volume_once(self):
        self.ngp.frame()
        dt = self.ngp.elapsed_training_time
        #print(f"Iter={self.total_iters}; Dt={dt}; Loss={self.ngp.loss}")
        if self.anneal and self.total_iters % self.anneal_every_iters == 0:
            self.ngp.nerf.training.depth_supervision_lambda *= self.annealing_rate
        if self.evaluate and self.total_iters % self.eval_every_iters == 0:
            print("Evaluate.")
            self.eval_gt_traj()
        self.total_iters += 1

    def evaluate_depth(self):
        self.mesh_renderer.render_depth()

    def print_ngp_info(self):
        print("NGP Info")
        ic(self.ngp.dynamic_res)
        ic(self.ngp.dynamic_res_target_fps)
        ic(self.ngp.fixed_res_factor)
        ic(self.ngp.background_color)
        ic(self.ngp.shall_train)
        ic(self.ngp.shall_train_encoding)
        ic(self.ngp.shall_train_network)
        ic(self.ngp.render_groundtruth)
        ic(self.ngp.groundtruth_render_mode)
        ic(self.ngp.render_mode)
        ic(self.ngp.slice_plane_z)
        ic(self.ngp.dof)
        ic(self.ngp.aperture_size)
        ic(self.ngp.autofocus)
        ic(self.ngp.autofocus_target)
        ic(self.ngp.floor_enable)
        ic(self.ngp.exposure)
        ic(self.ngp.scale)
        ic(self.ngp.bounding_radius)
        ic(self.ngp.render_aabb)
        ic(self.ngp.render_aabb_to_local)
        ic(self.ngp.aabb)
        ic(self.ngp.raw_aabb)
        ic(self.ngp.fov)
        ic(self.ngp.fov_xy)
        ic(self.ngp.fov_axis)
        ic(self.ngp.zoom)
        ic(self.ngp.screen_center)

    def print_training_info(self):
        print("Training Info")
        ic(self.ngp.nerf.training.n_images_for_training)
        ic(self.ngp.nerf.training.depth_supervision_lambda)

    def print_dataset_info(self):
        print("Dataset Info")
        ic(self.ngp.nerf.training.dataset.render_aabb)
        ic(self.ngp.nerf.training.dataset.render_aabb.min)
        ic(self.ngp.nerf.training.dataset.render_aabb.max)
        ic(self.ngp.nerf.training.dataset.render_aabb_to_local)
        ic(self.ngp.nerf.training.dataset.up)
        ic(self.ngp.nerf.training.dataset.offset)
        ic(self.ngp.nerf.training.dataset.n_images)
        ic(self.ngp.nerf.training.dataset.envmap_resolution)
        ic(self.ngp.nerf.training.dataset.scale)
        ic(self.ngp.nerf.training.dataset.aabb_scale)
        ic(self.ngp.nerf.training.dataset.from_mitsuba)
        ic(self.ngp.nerf.training.dataset.is_hdr)

    def print_dataset_metadata_info(self):
        print("Meta Info")
        metadatas = self.ngp.nerf.training.dataset.metadata
        ic(len(metadatas))
        for metadata in metadatas:
            ic(metadata.focal_length)
            ic(metadata.camera_distortion)
            ic(metadata.principal_point)
            ic(metadata.rolling_shutter)
            ic(metadata.light_dir)
            ic(metadata.resolution)

        ic(self.ngp.nerf.training.dataset.paths[0])
        #ic(self.ngp.nerf.training.dataset.transforms[0].start)
        #ic(self.ngp.nerf.training.dataset.transforms[0].end)

    def eval_gt_traj(self):
        ic(self.total_iters)

        spp = 1 # samples per pixel
        linear = True
        fps = 20.0

        # Save the state before evaluation
        import copy
        tmp_shall_train = copy.deepcopy(self.ngp.shall_train)
        tmp_background_color = copy.deepcopy(self.ngp.background_color)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_snap_to_pixel_centers = copy.deepcopy(self.ngp.snap_to_pixel_centers)
        tmp_rendering_min_transmittance = copy.deepcopy(self.ngp.nerf.rendering_min_transmittance)
        tmp_cam = self.ngp.camera_matrix.copy()
        tmp_render_mode = copy.deepcopy(self.ngp.render_mode)

        # Modify the state for evaluation
        self.ngp.background_color = [0.0, 0.0, 0.0, 1.0]
        self.ngp.snap_to_pixel_centers = True
        self.ngp.nerf.rendering_min_transmittance = 1e-4
        self.ngp.shall_train = False

        stride = 2

        # Evaluate
        count = 0
        total_l1 = 0
        total_psnr = 0
        assert(len(self.ref_frames) == self.ngp.nerf.training.n_images_for_training)
        for i in range(0, self.ngp.nerf.training.n_images_for_training, stride):
            # Use GT trajectory for evaluation to have consistent metrics.
            self.ngp.set_camera_to_training_view(i) 

            # Get ref/est RGB images
            self.ngp.render_mode = ngp.Shade
            ref_image = self.ref_frames[i][0]
            est_image = self.ngp.render(ref_image.shape[1], ref_image.shape[0], spp, linear, fps=fps)

            if self.viz:
                cv2.imshow("Color Error", np.sum(ref_image - est_image, axis=-1))

            # TODO: Get ref/est Depth images
            self.ngp.render_mode = ngp.Depth
            ref_depth = self.ref_frames[i][2].squeeze()
            est_depth = self.ngp.render(ref_image.shape[1], ref_image.shape[0], spp, linear, fps=fps)
            est_depth = est_depth[...,0] # The rest of the channels are the same (and last is 1)

            # Calc metrics
            mse = float(compute_error(est_image, ref_image))
            psnr = mse2psnr(mse)
            total_psnr += psnr

            # Calc L1 metrics
            if self.viz:
                frontend_depth = self.ref_frames[i][1].squeeze()
                depths_cov_up = torch.tensor(self.ref_frames[i][3], dtype=torch.float32, device="cpu")
                viz_depth_sigma(depths_cov_up.unsqueeze(0).sqrt(), fix_range=True,
                                bg_img=torch.tensor(ref_image[...,:3]*255, dtype=torch.uint8, device="cpu").permute(2,0,1).unsqueeze(0),
                                sigma_thresh=20.0, name="Depth Sigma")
                #import matplotlib.pyplot as plt
                #plt.hist(depths_cov_up.view(-1).cpu().numpy(), bins=50, density=False, histtype='barstacked',  # weights=weights_u,
                #        alpha=0.25, color=['steelblue'], edgecolor='none', label='cov', range=[0,256])
                #plt.legend(loc='upper right')
                #plt.xlabel('cov')
                #plt.ylabel('count')
                #plt.draw()
                #plt.pause(1)
                #plt.show()
                viz_depth_map(torch.tensor(frontend_depth, dtype=torch.float32, device="cpu"), fix_range=False, name="Frontend Depth", colormap=cv2.COLORMAP_TURBO, invert=False)
                viz_depth_map(torch.tensor(ref_depth, dtype=torch.float32, device="cpu"), fix_range=False, name="Ref Depth", colormap=cv2.COLORMAP_TURBO, invert=False)
                viz_depth_map(torch.tensor(est_depth, dtype=torch.float32, device="cpu"), fix_range=False, name="Est Depth", colormap=cv2.COLORMAP_TURBO, invert=False)

            est_to_ref_depth_scale = ref_depth.mean() / est_depth.mean()
            ic(est_to_ref_depth_scale)
            diff_depth_map = np.abs(est_to_ref_depth_scale * est_depth - ref_depth)
            diff_depth_map[diff_depth_map > 2.0] = 2.0 # Truncate outliers to 1m, otw biases metric, this can happen either bcs depth is not estimated or bcs gt depth is wrong. 
            if self.viz:
                viz_depth_map(torch.tensor(diff_depth_map), fix_range=False, name="Depth Error", colormap=cv2.COLORMAP_TURBO, invert=False)
            l1 = diff_depth_map.mean() * 100 # From m to cm AND use the mean (as in Nice-SLAM)
            total_l1 += l1
            count += 1

            if self.viz:
                ref_image_viz = cv2.cvtColor(ref_image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there
                est_image_viz = cv2.cvtColor(est_image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there
                cv2.imshow("Ref img", ref_image_viz)
                cv2.imshow("Est img", est_image_viz)

            if self.viz:
                cv2.waitKey(1)
            
        dt = self.ngp.elapsed_training_time
        psnr = total_psnr / (count or 1)
        l1 = total_l1 / (count or 1)
        print(f"Iter={self.total_iters}; Dt={dt}; PSNR={psnr}; L1={l1}; count={count}")
        self.df.loc[len(self.df.index)] = [self.total_iters, dt, psnr, l1, count]
        self.df.to_csv("results.csv")

        # Reset the state
        self.ngp.shall_train                 = tmp_shall_train
        self.ngp.background_color            = tmp_background_color
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.snap_to_pixel_centers       = tmp_snap_to_pixel_centers
        self.ngp.nerf.rendering_min_transmittance = tmp_rendering_min_transmittance
        self.ngp.camera_matrix               = tmp_cam
        self.ngp.render_mode                 = tmp_render_mode
