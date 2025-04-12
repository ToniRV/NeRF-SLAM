#!/usr/bin/env python3

from abc import abstractclassmethod

from collections import OrderedDict

from icecream import ic
import cv2

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from utils.flow_viz import *

import networks.geom.projective_ops as pops
from networks.modules.corr import CorrBlock, AltCorrBlock

import lietorch
from lietorch import SE3
import droid_backends

import gtsam
from gtsam import (HessianFactor)
from gtsam import Values
from gtsam import (Pose3, Rot3, Point3)
from gtsam import PriorFactorPose3
from gtsam import NonlinearFactorGraph
from gtsam import GaussianFactorGraph
from gtsam.symbol_shorthand import X

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def lietorch_pose_to_gtsam(pose : lietorch.SE3):
    trans, quat = pose.vec().split([3,4], -1)
    trans = trans.cpu().numpy()
    quat = quat.cpu().numpy()
    return Pose3(Rot3.Quaternion(quat[3], quat[0], quat[1], quat[2]), Point3(trans))

def gtsam_pose_to_torch(pose: gtsam.Pose3, device, dtype):
    t = pose.translation()
    q = pose.rotation().toQuaternion()
    return torch.tensor([t[0], t[1], t[2], q.x(), q.y(), q.z(), q.w()], device=device, dtype=dtype)

class VisualFrontend(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def forward(self, mini_batch):
        pass

from networks.modules.extractor import BasicEncoder
from networks.droid_net import UpdateModule

class RaftVisualFrontend(VisualFrontend):
    def __init__(self, world_T_body_t0, body_T_cam0, args, device="cpu"):
        super().__init__()
        self.args = args

        self.kf_idx = 0 # Keyframe index
        self.kf_idx_to_f_idx = {} # Keyframe index to frame index 
        self.f_idx_to_kf_idx = {} # Frame index to keyframe index
        self.last_kf_idx = 0
        self.last_k = None

        self.global_ba = False

        self.stop = False # stop module

        self.compute_covariances = True

        self.last_state = gtsam.Values()

        self.initial_x0 = None
        self.initial_priors = None

        self.factors_to_remove = gtsam.KeyVector()

        self.buffer = args.buffer
        self.stereo = args.stereo
        self.device = device

        self.is_initialized = False

        self.keyframe_warmup = 8
        self.max_age = 25
        self.max_factors = 48
        self.kf_init_count = 8
        self.motion_filter_thresh = 2.4 # To determine if we are moving, how much mean optical flow before considering new frame [px]

        self.viz = False # Whether to visualize the results

        self.world_T_body_t0 = world_T_body_t0
        self.body_t0_T_world = gtsam_pose_to_torch(self.world_T_body_t0.inverse(), self.device, torch.float)
        self.body_T_cam0 = body_T_cam0
        self.world_T_cam0_t0 = world_T_body_t0 * body_T_cam0
        self.cam0_t0_T_world = gtsam_pose_to_torch(
            self.world_T_cam0_t0.inverse(), self.device, torch.float)
        self.cam0_T_body = gtsam_pose_to_torch(
            body_T_cam0.inverse(), self.device, torch.float)

        # Frontend params
        self.keyframe_thresh = 4.0 # Distance to consider a keyframe, threshold to create a new keyframe [m] # why not 0.4!
        self.frontend_thresh = 16.0 # Add edges between frames within this distance
        self.frontend_window = 25 # frontend optimization window
        self.frontend_radius = 2 # force edges between frames within radius
        self.frontend_nms    = 1 # non-maximal supression of edges
        self.beta            = 0.3 # weight for translation / rotation components of flow # also used in backend

        # Backend params
        self.backend_thresh = 22.0
        self.backend_radius = 2
        self.backend_nms = 3

        self.iters1 = 4 # number of iterations for first optimization
        self.iters2 = 2 # number of iterations for second optimization

        # DownSamplingFactor: resolution of the images with respect to the features extracted.
        # 8.0 means that the features are at 1/8th of the original resolution.
        self.dsf = 8 # perhaps the most important parameter

        # Type of correlation computation to use: "volume" or "alt"
        # "volume" takes a lot of memory (but is faster), "alt" takes less memory and should be as fast as volume but it's not
        self.corr_impl = "volume"

        # Build Networks
        self.feature_net = BasicEncoder(output_dim=128, norm_fn='instance')
        self.context_net = BasicEncoder(output_dim=256, norm_fn='none')
        self.update_net = UpdateModule()

        # Load network weights
        weights = self.load_weights(args.weights)
        missing_keys = self.load_state_dict(weights)
        self.to(device)
        self.eval()

        # Uncertainty sigmas, initial sigmas for initialization (but not priors?)
        self.translation_sigma = torch.tensor(0.01, device=self.device) # standard deviation of translation [m]
        self.rotation_sigma = torch.tensor(0.01, device=self.device) # standard deviation of rotation [rad]
        # TODO: given that the values are much larger than 1.0... we should increase this much more...
        self.sigma_idepth = torch.tensor(0.1, device=self.device) # standard deviation of depth [m] (or inverse depth?) [1/m], we don't know the scale anyway...
        self.t_cov = torch.pow(self.translation_sigma, 2) * torch.eye(3, device=self.device)
        self.r_cov = torch.pow(self.rotation_sigma, 2) * torch.eye(3, device=self.device)
        self.idepth_prior_cov = torch.pow(self.sigma_idepth, 2)
        self.g_prior_cov = torch.block_diag(self.r_cov, self.t_cov) # GTSAM convention, rotation first, then translation

    def __del__(self):
        print("Calling frontend dtor...")
        torch.cuda.empty_cache()

    def stop_condition(self):
        return self.stop

    # Pre-allocate all the memory in the GPU
    def initialize_buffers(self, image_size):
        self.img_height = h = image_size[0]
        self.img_width  = w = image_size[1]

        ic(self.dsf)
        ic(h)
        ic(w)
        ic(h//self.dsf)
        ic(w//self.dsf)

        self.coords0 = pops.coords_grid(h//self.dsf, w//self.dsf, device=self.device)
        self.ht, self.wd = self.coords0.shape[:2]

        ### Input attributes ###
        self.cam0_timestamps = torch.zeros(self.buffer, dtype=torch.float, device=self.device).share_memory_()
        # TODO this should be in the euroc parser, so that we don't allocate memory without bounds
        self.cam0_images     = torch.zeros(self.buffer, 3, h, w, dtype=torch.uint8, device=self.device).share_memory_() # TODO why not shared memory? # This is a looot of memory
        self.cam0_intrinsics = torch.zeros(self.buffer, 4,       dtype=torch.float, device=self.device).share_memory_()
        self.gt_poses        = torch.zeros(self.buffer, 4, 4,    dtype=torch.float, device=self.device).share_memory_()
        self.gt_depths       = torch.zeros(self.buffer, 1, h, w, dtype=torch.float, device=self.device).share_memory_()

        ### State attributes ###
        self.cam0_T_world          = torch.zeros(self.buffer, 7,   dtype=torch.float, device=self.device).share_memory_()
        self.world_T_body        = torch.zeros(self.buffer, 7,   dtype=torch.float, device=self.device).share_memory_()
        self.world_T_body_cov    = torch.zeros(self.buffer, 6, 6,   dtype=torch.float, device=self.device).share_memory_()
        self.cam0_idepths        = torch.ones(self.buffer,  h//self.dsf, w//self.dsf, dtype=torch.float, device=self.device).share_memory_()
        self.cam0_idepths_cov    = torch.ones(self.buffer,  h//self.dsf, w//self.dsf, dtype=torch.float, device=self.device).share_memory_()
        self.cam0_depths_cov     = torch.ones(self.buffer,  h//self.dsf, w//self.dsf, dtype=torch.float, device=self.device).share_memory_()
        self.cam0_idepths_sensed = torch.zeros(self.buffer, h//self.dsf, w//self.dsf, dtype=torch.float, device=self.device).share_memory_()
        self.cam0_idepths_up     = torch.zeros(self.buffer, h, w, dtype=torch.float, device=self.device).share_memory_() # This is a looot of memory
        self.cam0_depths_cov_up  = torch.ones(self.buffer,  h, w, dtype=torch.float, device=self.device).share_memory_() # This is a looot of memory

        # INITIALIZE state:
        # - poses all to initial state transformation
        # - velocities all to 0 except first to initial state
        # - biases all to 0 except first to initial state
        # TODO: why not shared memory?
        self.cam0_T_world[:]       = self.cam0_t0_T_world
        self.world_T_body[:]     = gtsam_pose_to_torch(self.world_T_body_t0, device=self.device, dtype=torch.float)
        self.world_T_body_cov[:] = self.g_prior_cov * torch.eye(6, device=self.device) #* torch.as_tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001], device=self.device)[None]
        self.cam0_idepths_cov   *= self.idepth_prior_cov

        # For multi-view, we could set this to >2, but we need to know overlapping FOVs
        # we could do that automatically by looking at the mean flow between frames
        cameras = 2 if self.stereo else 1

        ### Feature attributes ### Every keyframe has a feature/context/gru_input
        self.features_imgs     = torch.zeros(self.buffer, cameras, 128, h//self.dsf, w//self.dsf, dtype=torch.half, device=self.device)#.share_memory_()
        self.contexts_imgs     = torch.zeros(self.buffer, cameras, 128, h//self.dsf, w//self.dsf, dtype=torch.half, device=self.device)#.share_memory_()
        self.cst_contexts_imgs = torch.zeros(self.buffer, cameras, 128, h//self.dsf, w//self.dsf, dtype=torch.half, device=self.device)#.share_memory_()

        ### Correlations, Flows, and Hidden States ### Every pair of co-visible keyframes has a correlation volume, flow, and hidden state
        # These are created on-the-fly so we can't really pre-allocate memory
        self.correlation_volumes       = None
        self.gru_hidden_states         = None # initialized as context, but evolves as hidden state
        self.gru_contexts_input        = None # initialized as context, and remains as such
        self.gru_estimated_flow        = torch.zeros([1, 0, h//self.dsf, w//self.dsf, 2], device=self.device, dtype=torch.float)
        self.gru_estimated_flow_weight = torch.zeros([1, 0, h//self.dsf, w//self.dsf, 2], device=self.device, dtype=torch.float)
        self.damping = 1e-6 * torch.ones_like(self.cam0_idepths) # not sure what this does

        ### Co-visibility Graph ###
        self.ii  = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj  = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.age = torch.as_tensor([], dtype=torch.long, device=self.device)

        # inactive factors
        self.ii_inactive = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_inactive = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.ii_bad  = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_bad  = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.gru_estimated_flow_inactive   = torch.zeros([1, 0, h//self.dsf, w//self.dsf, 2], device=self.device, dtype=torch.float)
        self.gru_estimated_flow_weight_inactive = torch.zeros([1, 0, h//self.dsf, w//self.dsf, 2], device=self.device, dtype=torch.float)

        # For visualization, True: needs viz update, False: not changed
        self.viz_idx = torch.zeros(self.buffer, device=self.device, dtype=torch.bool)

    #@abstractclassmethod
    def forward(self, batch):
        # The output of RaftVisualFrontend is not dense optical flow
        # but rather a bunch of pose-to-pose factors resulting from the reduced camera matrix.
        print("RaftVisualFrontend.forward")

        k = batch["k"][0]

        # The output
        x0 = Values()# None # 
        factors = NonlinearFactorGraph()# None # 
        viz_out = None

        imgs_k = torch.as_tensor(batch["images"], device=self.device)[None].permute(0, 1, 4, 2, 3)#.shared_memory()
        imgs_norm_k = self._normalize_imgs(imgs_k)

        if self.viz:
            for i, img in enumerate(imgs_k[0]):
                cv2.imshow(f'Img{i} input', img.permute(1,2,0).cpu().numpy())
            for i, img in enumerate(imgs_norm_k[0]):
                cv2.imshow(f'Img{i} normalized', img.permute(1,2,0).cpu().numpy())

        if self.last_k is None:
            ic(k)
            assert k == 0
            assert self.kf_idx == 0
            assert self.last_kf_idx == 0

            # Initialize network buffers
            self.initialize_buffers(imgs_k.shape[-2:]) # last two dims are h,w
            self.gt_poses[self.kf_idx]        = torch.tensor(batch["poses"][0],           device=self.device)
            if batch["depths"][0] is not None:
                self.gt_depths[self.kf_idx]   = torch.tensor(batch["depths"][0],          device=self.device).permute(2,0,1)
            self.cam0_timestamps[self.kf_idx] = torch.tensor(batch["t_cams"][0],          device=self.device)
            self.cam0_images[self.kf_idx]     = torch.tensor(batch["images"][0],          device=self.device)[..., :3].permute(2,0,1)
            self.cam0_intrinsics[self.kf_idx] = (1.0 / self.dsf) * torch.tensor(batch["calibs"][0].camera_model.numpy(), device=self.device)

            # Initialize the state
            # Compute its dense features for next iteration
            self.features_imgs[self.kf_idx] = self.__feature_encoder(imgs_norm_k)
            # Compute its context features for next iteration
            self.contexts_imgs[self.kf_idx], self.cst_contexts_imgs[self.kf_idx] = self.__context_encoder(imgs_norm_k)

            # Store things for next iteration
            self.last_k                       = k
            self.last_kf_idx                  = self.kf_idx
            self.kf_idx_to_f_idx[self.kf_idx] = k
            self.f_idx_to_kf_idx[k]           = self.kf_idx
            viz_out = self.get_viz_out(batch)
            self.kf_idx                      += 1
            return x0, factors, viz_out

        assert k > 0
        assert self.kf_idx < self.buffer

        # Add frame as keyframe if we have enough motion, otherwise discard:
        current_imgs_features = self.__feature_encoder(imgs_norm_k)
        if not self.has_enough_motion(current_imgs_features):
            if batch["is_last_frame"]:
                self.kf_idx -= 1 # Because in the last iter we increased it, but aren't taking any...
                print("Last frame reached, and no new motion: starting GLOBAL BA")
                self.terminate()
                # Send the whole viz_out to update the fact that BA has changed all poses/depths
                viz_out = self.get_viz_out(batch)
                return x0, factors, viz_out
            # By returning, we do not increment self.kf_idx
            return x0, factors, viz_out

        # Ok, we got enough motion, consider this frame as a keyframe
        # Compute dense optical flow
        self.gt_poses[self.kf_idx]        = torch.tensor(batch["poses"][0],           device=self.device)
        if batch["depths"][0] is not None:
            self.gt_depths[self.kf_idx]   = torch.tensor(batch["depths"][0],          device=self.device).permute(2,0,1)
        self.cam0_timestamps[self.kf_idx] = torch.tensor(batch["t_cams"][0],          device=self.device)
        self.cam0_images[self.kf_idx]     = torch.tensor(batch["images"][0],          device=self.device)[..., :3].permute(2,0,1)
        self.cam0_intrinsics[self.kf_idx] = (1.0 / self.dsf) * torch.tensor(batch["calibs"][0].camera_model.numpy(), device=self.device)
        self.features_imgs[self.kf_idx]   = current_imgs_features
        self.contexts_imgs[self.kf_idx], self.cst_contexts_imgs[self.kf_idx] = self.__context_encoder(imgs_norm_k)
        self.kf_idx_to_f_idx[self.kf_idx] = k
        self.f_idx_to_kf_idx[k]           = self.kf_idx

        # Build the flow graph: ii -> jj edges
        # TODO: for now just do a chain
        # Just adds the `r' sequential frames to the graph
        # do initialization
        if not self.is_initialized:
            if self.kf_idx >= self.keyframe_warmup:
                self.__initialize()
            else:
                # We don't return here so that we increment kf_idx
                # ic("Warming up: pre-processing frame with enough motion")
                pass
        # do update
        else:
            if not self.__update():
                # We did not accept this keyframe, reinit its properties (really needed? they will be overwritten no?)
                # Remove as well factors connected to it used for estimating its distance...
                self.rm_keyframe(self.kf_idx - 1) # TODO: the -1 here changed the whole behavior, check if it is correct
                # Decrease kf_idx since we are literally removing the keyframe...
                # But this means that we need to keep track the difference btw the kf_idx in the backend
                # and the keyframe_idx in the frontend.
                #self.video.counter.value -= 1
                # self.kf_idx -= 1 # so that on the next pass we use the previous keyframe
                # By returning, we do not increment self.kf_idx
                return x0, factors, viz_out

        #x0.insert(X(k), pose_to_gtsam(last_pose))

        self.last_k                       = k
        self.last_kf_idx                  = self.kf_idx
        self.kf_idx_to_f_idx[self.kf_idx] = k # not really necessary I think
        self.f_idx_to_kf_idx[k]           = self.kf_idx

        viz_out = self.get_viz_out(batch) # build viz_out after updating self.kf_idx_to_f_idx

        if self.viz:
            cv2.waitKey(1)

        if self.kf_idx + 1 >= self.buffer or batch["is_last_frame"]:
            print("Buffer full or last frame reached: starting GLOBAL BA")
            self.terminate()
            viz_out = self.get_viz_out(batch)
            return x0, factors, viz_out

        self.kf_idx += 1

        return x0, factors, viz_out


    # If kf0 is None, then it is init to keyframe number 1 or min(ii)+1, TODO: why not 0?
    # If kf1 is None, then it is init to the max(ii, jj) +1, TODO: again, why the +1?
    @torch.cuda.amp.autocast(enabled=True)
    def update(self, kf0=None, kf1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        #ic("Memory usage before update: {} Mb".format(torch.cuda.memory_allocated()/1024**2))

        # motion features
        with torch.cuda.amp.autocast(enabled=False): # try mixed precision?
            # Coords1 shape is: (batch_size, num_edges, ht, wd, 2)
            coords1, mask, (Ji, Jj, Jz) = self.reproject(self.ii, self.jj, cam_T_body=self.cam0_T_body, jacobian=True) # this is not using the cuda kernels... # mask is not used...
            # "coords1 - coords0": coords1 = coords0 + flow, so this is the current
            # flow induced by the estimated pose/depth.
            # "target - coords1": residual from current
            # - `measured` flow (target) by GRU: target = coords1 + flow_delta (from GRU), and
            # - `estimated' flow (coords1): coords1 = coords0 + reproject()
            motion = torch.cat([coords1 - self.coords0, self.gru_estimated_flow - coords1], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        # correlation features
        corr = self.correlation_volumes(coords1)

        # We then pool the hidden state over all features which share the same source view i and predict a
        # pixel-wise damping factor λ. We use the softplus operator to ensure that the damping term is positive.
        # Additionally, we use the pooled features to predict a 8x8 mask which can be used to upsample the
        # inverse depth estimate
        self.gru_hidden_states, flow_delta, gru_estimated_flow_weight, damping, upmask = \
            self.update_net(self.gru_hidden_states, self.gru_contexts_input,
                            corr, flow=motion, ii=self.ii, jj=self.jj)

        if kf0 is None:
            # In droid: kf0 = max(1, self.ii.min().item()+1) # It is max(1, min(ii)) because the first pose is fixed.
            kf0 = max(0, self.ii.min().item()) 
        else:
            ic(kf0)
            raise

        with torch.cuda.amp.autocast(enabled=False):
            # flow_delta shape is: (batch_size, num_edges, ht, wd, 2)
            self.gru_estimated_flow        = coords1 + flow_delta.to(dtype=torch.float)
            self.gru_estimated_flow_weight = gru_estimated_flow_weight.to(dtype=torch.float)

            self.damping[torch.unique(self.ii)] = damping # TODO What is this damping? See also `damping` below

            if use_inactive: # It is always set to True...
                # TODO What is this doing? I think this is somehow setting the priors!! Or
                # the marginalization priors... 
                # The thing is that, there are two considerations:
                # - depth_maps to optimize over (K = len(torch.unique(ii)))
                # - keyframe poses to optimize over (P = kf1-kf0)
                # where K > P, so that there are some poses we do not optimize over (but we do optimize their depth-maps)
                mask = (self.ii_inactive >= kf0 - 3) & (self.jj_inactive >= kf0 - 3)
                ii = torch.cat([self.ii_inactive[mask], self.ii], 0)
                jj = torch.cat([self.jj_inactive[mask], self.jj], 0)
                gru_estimated_flow        = torch.cat([self.gru_estimated_flow_inactive[:,mask], self.gru_estimated_flow], 1)
                gru_estimated_flow_weight = torch.cat([self.gru_estimated_flow_weight_inactive[:,mask], self.gru_estimated_flow_weight], 1)
            else:
                ii, jj, gru_estimated_flow, gru_estimated_flow_weight = self.ii, self.jj, self.gru_estimated_flow, self.gru_estimated_flow_weight

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP # TODO What is this damping?

            # gru_estimated_flow(_weight) shape after this line is: (num_edges, 2, ht, wd)
            gru_estimated_flow        = gru_estimated_flow.view(-1, self.ht, self.wd, 2).permute(0,3,1,2).contiguous()
            gru_estimated_flow_weight = gru_estimated_flow_weight.view(-1, self.ht, self.wd, 2).permute(0,3,1,2).contiguous()

            # TODO We should output at this point the GRU estimated flow and weights.
            # Or rather the factors?

            # Dense bundle adjustment
            ic("BA!")
            x0, rcm_factor = self.ba(gru_estimated_flow, gru_estimated_flow_weight, damping,
                                 ii, jj, kf0, kf1, itrs=itrs, lm=1e-4, ep=0.1,
                                 motion_only=motion_only, compute_covariances=self.compute_covariances)

            # Stores depths_up, depths_cov_up
            kx = torch.unique(self.ii)
            self.cam0_idepths_up[kx] = cvx_upsample(self.cam0_idepths[kx].unsqueeze(-1), upmask).squeeze()
            self.cam0_depths_cov_up[kx] = cvx_upsample(self.cam0_depths_cov[kx].unsqueeze(-1), upmask, pow=1.0).squeeze()

            if self.viz:
                viz_idepth(self.cam0_idepths[kx], upmask)
                viz_idepth_sigma(self.cam0_idepths_cov[kx], upmask, fix_range=True, bg_img=self.cam0_images[kx])
                viz_depth_sigma(self.cam0_depths_cov_up[kx].unsqueeze(-1).sqrt(), fix_range=True, bg_img=self.cam0_images[kx], sigma_thresh=20.0)
                viz_flow("gru_flow", gru_estimated_flow[-1] - self.coords0.permute(2,0,1))
                reprojection_flow = (coords1 - self.coords0).squeeze().permute(0,3,1,2)
                viz_flow("reprojection_flow", reprojection_flow[-1])

            # Viz weight
            if self.viz:
                # Visualize input image as well
                #for k, i in enumerate(self.ii):
                #    self.viz_weight(gru_estimated_flow_weight[k], self.cam0_images[ii[k]]) #self.ii[-1].item(), self.jj[-1].item())
                self.viz_weight(gru_estimated_flow_weight[-1], self.cam0_images[ii[-1]]) #self.ii[-1].item(), self.jj[-1].item())

            # Update visualization
            kf1 = max(ii.max().item(), jj.max().item())
            assert kf1 == self.kf_idx
            self.viz_idx[kf0:self.kf_idx+1] = True
            
        self.age += 1

        return x0, rcm_factor


    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        kfs, cameras, ch, ht, wd = self.features_imgs.shape
        corr_op = AltCorrBlock(self.features_imgs.view(1, kfs*cameras, ch, ht, wd))

        for step in range(steps):
            print(f"Global BA Iteration #{step}/{steps}")
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask, _ = self.reproject(self.ii, self.jj)
                motion = torch.cat([coords1 - self.coords0, self.gru_estimated_flow - coords1], dim=-1)
                motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            # CONVGRU RUNS
            # Optimize the flow as much as possible, 
            s = 8
            for i in range(0, self.jj.max() + 1, s): # what does this do?
                print(f"ConvGRU Iteration #{i/s}/{(self.jj.max() + 1)/s}")
                v = (self.ii >= i) & (self.ii < i + s) # kind-of like a sliding optimization window
                iis = self.ii[v]
                jjs = self.jj[v]

                corr = corr_op(coords1[:,v], cameras * iis, cameras * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                    # TODO: somehow the damping and upmask have weird shapes... what is going on?
                    gru_hidden_states, flow_delta, gru_estimated_flow_weight, damping, upmask = \
                        self.update_net(self.gru_hidden_states[:, v], self.gru_contexts_input[:, iis], corr, motion[:, v], iis, jjs)

                kx = torch.unique(iis)
                all_kf_ids = torch.unique(torch.cat([iis, jjs], 0))

                self.gru_hidden_states[:,v]         = gru_hidden_states
                self.gru_estimated_flow[:,v]        = coords1[:,v] + flow_delta.float()
                self.gru_estimated_flow_weight[:,v] = gru_estimated_flow_weight.float()
                self.damping[all_kf_ids] = damping # TODO What is this damping? See also `damping` below

                # Stores depths_up, depths_cov_up
                self.cam0_idepths_up[all_kf_ids]    = cvx_upsample(self.cam0_idepths[all_kf_ids].unsqueeze(-1), upmask).squeeze()
                self.cam0_depths_cov_up[all_kf_ids] = cvx_upsample(self.cam0_depths_cov[all_kf_ids].unsqueeze(-1), upmask, pow=1.0).squeeze()

            #ii = torch.cat([self.ii_inactive[mask], self.ii], 0)
            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            gru_estimated_flow        = self.gru_estimated_flow.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            gru_estimated_flow_weight = self.gru_estimated_flow_weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            ic("Global BA!")
            # TODO: do not compute cov for global BA until we fix the memory issue when building Eiz
            x0, rcm_factor = self.ba(gru_estimated_flow, gru_estimated_flow_weight, damping, self.ii, self.jj,
                                     kf0=0, kf1=None, itrs=itrs, lm=1e-5, ep=1e-2,
                                     motion_only=False, compute_covariances=False)


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, kf_idx):
        """ drop nodes from factor graph """

        # TODO: how does this work if we rm a keyframe that is not the one before the last one??
        # As of now, kf_idx is the last one, and so kf_idx+1 is all 0s.
        self.gt_poses[kf_idx]           = self.gt_poses[kf_idx+1]
        self.gt_depths[kf_idx]          = self.gt_depths[kf_idx+1]
        self.cam0_images[kf_idx]        = self.cam0_images[kf_idx+1]
        self.cam0_timestamps[kf_idx]    = self.cam0_timestamps[kf_idx+1]
        self.cam0_T_world[kf_idx]       = self.cam0_T_world[kf_idx+1]
        self.world_T_body[kf_idx]       = self.world_T_body[kf_idx+1]
        self.world_T_body_cov[kf_idx]   = self.world_T_body_cov[kf_idx+1]
        self.cam0_idepths[kf_idx]       = self.cam0_idepths[kf_idx+1]
        self.cam0_idepths_cov[kf_idx]   = self.cam0_idepths_cov[kf_idx+1]
        self.cam0_depths_cov[kf_idx]    = self.cam0_depths_cov[kf_idx+1]
        self.cam0_idepths_sensed[kf_idx] = self.cam0_idepths_sensed[kf_idx+1]
        self.cam0_intrinsics[kf_idx]    = self.cam0_intrinsics[kf_idx+1]

        self.features_imgs[kf_idx]     = self.features_imgs[kf_idx+1]
        self.contexts_imgs[kf_idx]     = self.contexts_imgs[kf_idx+1]
        self.cst_contexts_imgs[kf_idx] = self.cst_contexts_imgs[kf_idx+1]

        # Remove all inactive edges that are connected to the keyframe
        mask = (self.ii_inactive == kf_idx) | (self.jj_inactive == kf_idx)

        # Reindex the inactive edges that we are going to keep
        self.ii_inactive[self.ii_inactive >= kf_idx] -= 1
        self.jj_inactive[self.jj_inactive >= kf_idx] -= 1

        # Remove the inactive edges concerning this keyframe
        if torch.any(mask):
            self.ii_inactive     = self.ii_inactive[~mask]
            self.jj_inactive     = self.jj_inactive[~mask]
            self.gru_estimated_flow_inactive = self.gru_estimated_flow_inactive[:,~mask]
            self.gru_estimated_flow_weight_inactive = self.gru_estimated_flow_weight_inactive[:,~mask]

        # Remove all edges that are connected to the keyframe
        mask = (self.ii == kf_idx) | (self.jj == kf_idx)

        # Reindex the edges that we are going to keep
        self.ii[self.ii >= kf_idx] -= 1
        self.jj[self.jj >= kf_idx] -= 1

        # Remove the data concerning this keyframe (correlation volumes, etc.)
        self.rm_factors(mask, store=False)


    def __update(self):
        """ add edges, perform update """

        # self.count += 1 # TODO
        #self.kf_idx += 1  # TODO I think this is our kf_idx

        if self.correlation_volumes is not None:
            # Really only drops edges...
            #ic("Removing factors, and storing.")
            self.rm_factors(self.age > self.max_age, store=True)

        # TODO: unclear how this works 
        # t = self.kf_idx
        # ix = torch.arange(kf0, t)
        # jx = torch.arange(kf1, t)
        #ic("Adding proximity factors")
        self.add_proximity_factors(kf0=self.kf_idx - 4,
                                   kf1=max(self.kf_idx + 1 - self.frontend_window, 0),
                                   rad=self.frontend_radius, nms=self.frontend_nms,
                                   thresh=self.frontend_thresh, beta=self.beta, remove=True)

        # Initialize current cam0_depths with the sensed depths if valid,
        # otherwise with the previous mean depth (during init it is the mean amongst last 4 frames) 
        # (perhaps try with the previous raw depth)
        self.cam0_idepths[self.kf_idx] = torch.where(self.cam0_idepths_sensed[self.kf_idx] > 0,
                                                    self.cam0_idepths_sensed[self.kf_idx],
                                                    self.cam0_idepths[self.kf_idx])
        # TODO: should we lower the sigmas of idepth here since it is a measurement

        #ic("First update")
        for itr in range(self.iters1):
            x0, rcm_factor= self.update(kf0=None, kf1=None, use_inactive=True)

        # Get distance between the previous keyframe (kf_idx-2) and the current frame (kf_idx-1)
        d = self.distance([self.kf_idx-2], [self.kf_idx-1], beta=self.beta, bidirectional=True)

        #ic(d.item())
        if d.item() < self.keyframe_thresh:
            #ic("Not a keyframe")
            return False
        else:
            #ic("Found a keyframe")
            #ic("Second update")
            for itr in range(self.iters2):
                x0, rcm_factor = self.update(None, None, use_inactive=True)

            # TODO: I believe this should be inside this conditional rather than outside,
            # Because in the previous case we decided not to accept the keyframe...
            # set pose for next iteration
            next_kf = self.kf_idx + 1  
            if next_kf < self.buffer: # if we have not reached the end of the buffer (aka end of sequence)
                self.cam0_T_world[next_kf]         = self.cam0_T_world[self.kf_idx]
                self.world_T_body[next_kf]       = self.world_T_body[self.kf_idx]
                self.world_T_body_cov[next_kf]   = self.world_T_body_cov[self.kf_idx]
                # Why not just the previous depths as in init?
                #self.cam0_idepths[next_kf] = self.cam0_idepths[self.kf_idx]
                self.cam0_idepths[next_kf]      = self.cam0_idepths[self.kf_idx].mean()
                self.cam0_idepths_cov[next_kf]  = self.cam0_idepths_cov[self.kf_idx]
                self.cam0_depths_cov[next_kf]   = self.cam0_depths_cov[self.kf_idx]
                #self.viz_idx[next_kf] = True

            return True


    def __initialize(self):
        """ initialize the SLAM system """
        assert(self.kf_idx > 4)
        assert(self.kf_idx >= self.keyframe_warmup)

        # Just adds the `radius' sequential frames to the graph
        self.add_neighborhood_factors(kf0=0, kf1=self.kf_idx, radius=3)

        for _ in range(8):
            x0, rcm_factor = self.update(kf0=None, kf1=None, use_inactive=True)

        # Adds factors between frames, but unsure how this actually works...
        # if kf0 and kf1 are 0, then it tries to add prox factors btw all keyframes
        # t = self.kf_idx
        # ix = torch.arange(kf0, t)
        # jx = torch.arange(kf1, t)
        #ic("Add proximity factors")
        self.add_proximity_factors(kf0=0, kf1=0, rad=2, nms=2,
                                   thresh=self.frontend_thresh, remove=False)

        for _ in range(8):
            x0, rcm_factor = self.update(kf0=None, kf1=None, use_inactive=True)

        # TODO: next kf_idx shouldn't be kf_idx+1?
        # Set initial pose/depth for next iteration
        self.cam0_T_world[self.kf_idx + 1]      = self.cam0_T_world[self.kf_idx].clone()
        self.world_T_body[self.kf_idx + 1]      = self.world_T_body[self.kf_idx].clone()
        self.world_T_body_cov[self.kf_idx + 1]  = self.world_T_body_cov[self.kf_idx].clone()
        # TODO: Next depth is just the mean of the previous 4 depths?
        # We just retrieve our global implicit map here....
        self.cam0_idepths[self.kf_idx + 1]      = self.cam0_idepths[self.kf_idx - 3:self.kf_idx+1].mean()
        # TODO: here we are doing something very wrong and it is to previous sigmas and worst the initial sigma which is just one
        # because I don't have a better initialization... but looking at the numbers one is very confident...
        self.cam0_idepths_cov[self.kf_idx + 1]  = self.cam0_idepths_cov[self.kf_idx-3:self.kf_idx+1].mean()
        self.cam0_depths_cov[self.kf_idx + 1]   = self.cam0_depths_cov[self.kf_idx-3:self.kf_idx+1].mean()

        # initialization complete
        self.is_initialized = True

        # Update visualization
        self.viz_idx[:self.kf_idx+1] = True

        # TODO: what is this 4 here? What if warmup is less than 4...
        # Remove edges that point to the first keyframes...
        #ic("Remove factors after init, and storing")
        self.rm_factors(self.ii < (self.keyframe_warmup - 4), store=True)

        return x0, rcm_factor

    def add_neighborhood_factors(self, kf0, kf1, radius=3):
        """ add edges between neighboring frames within radius r """

        # Build dense adjacency matrix, +1 bcs arange stops but does not include the last index.
        ii, jj = torch.meshgrid(torch.arange(kf0, kf1+1), torch.arange(kf0, kf1+1))
        ii     = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj     = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.stereo else 0

        # Remove from dense adjacency matrix those that are not within `r' frames of each other
        # This basically keeps the `r' sub-diagonals at the top/bottom of the diagonal.
        distances = torch.abs(ii - jj)
        keep_radius = distances <= radius
        keep_stereo = distances > c # TODO: not sure why this is like this...
        keep = keep_stereo & keep_radius

        # Computes the correlation between these frames
        self.add_factors(ii[keep], jj[keep])


    # TODO: not sure how this really works...
    def add_proximity_factors(self, kf0=0, kf1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.kf_idx + 1
        ix = torch.arange(kf0, t)
        jx = torch.arange(kf1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.distance(ii, jj, beta=beta)
        d[(ii - rad) < jj] = np.inf # Set closer than rad frames distance to infinity
        d[d > 100] = np.inf # Set distances greater than 100 to infinity

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inactive], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inactive], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (kf0 <= i1 < t) and (kf1 <= j1 < t):
                            d[(i1-kf0)*(t-kf1) + (j1-kf1)] = np.inf

        es = []
        for i in range(kf0, t):
            if self.stereo:
                es.append((i, i))
                d[(i-kf0)*(t-kf1) + (i-kf1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-kf0)*(t-kf1) + (j-kf1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]

            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (kf0 <= i1 < t) and (kf1 <= j1 < t):
                            d[(i1-kf0)*(t-kf1) + (j1-kf1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)


    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """
        return_distance_matrix = False
        if ii is None:
            return_distance_matrix = True
            N = self.kf_idx
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii, jj = self.format_indicies(ii, jj, device=self.device)

        if bidirectional:
            poses = self.cam0_T_world[:self.kf_idx+1].clone() # TODO: why clone? why +1?
            d1 = droid_backends.frame_distance(poses, self.cam0_idepths, self.cam0_intrinsics[0], ii, jj, beta)
            d2 = droid_backends.frame_distance(poses, self.cam0_idepths, self.cam0_intrinsics[0], jj, ii, beta)
            d = .5 * (d1 + d2)
        else:
            d = droid_backends.frame_distance(self.cam0_T_world, self.cam0_idepths, self.cam0_intrinsics[0], ii, jj, beta)

        if return_distance_matrix:
            return d.reshape(N, N)

        return d

    # Computes correlations for each edge in (ii, jj)
    # Inits gru_hidden_states for each edge from self.contexts_imgs
    # Computes the gru_context_input from the cst_contexts_imgs
    # Inits gru_estimated_flow to the reprojected flow
    # Inits gru_estimated_flow_weight to 0.0
    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges, duplication happens because neighborhood and proximity factors
        # may overlap.
        ii, jj = self.__filter_repeated_edges(ii, jj)
        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        old_factors_count = self.ii.shape[0]
        new_factors_count = ii.shape[0]
        if self.max_factors > 0 and \
           old_factors_count + new_factors_count > self.max_factors \
           and self.correlation_volumes is not None and remove:
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            #ic("Remove old factors, and storing.")
            self.rm_factors(ix >= (self.max_factors - new_factors_count), store=True)

        ### Add new factors
        self.ii  = torch.cat([self.ii, ii], 0)
        self.jj  = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        ### Add correlation volumes for new edges, if we do not use alt implementation
        # that computes correlations on the fly... (a bit slower for some reason, but saves
        # tons of memory, since most of the corr volume will be unexplored I think)
        if self.corr_impl == "volume":
            is_stereo = (ii == jj).long() # TODO: does this still hold for multi-camera?
            feature_img_ii = self.features_imgs[None, ii, 0]
            feature_img_jj = self.features_imgs[None, jj, is_stereo]
            corr = CorrBlock(feature_img_ii, feature_img_jj)
            self.correlation_volumes = self.correlation_volumes.cat(corr) \
                if self.correlation_volumes is not None else corr

        ### Gru hidden states are initialized to the context features, and then they evolve
        gru_hidden_state = self.contexts_imgs[None, ii, 0] # Only for cam0
        self.gru_hidden_states = torch.cat([self.gru_hidden_states, gru_hidden_state], 1) \
            if self.gru_hidden_states is not None else gru_hidden_state
        ### Gru input states are initialized to the context features, and they do not evolve
        gru_context_input = self.cst_contexts_imgs[None, ii, 0] # Only for cam0
        self.gru_contexts_input = torch.cat([self.gru_contexts_input, gru_context_input], 1) \
            if self.gru_contexts_input is not None else gru_context_input

        ### Gru estimated flow is initialized to the reprojected flow, and then it evolves
        with torch.cuda.amp.autocast(enabled=False):
            target, _, _ = self.reproject(ii, jj)
            weight = torch.zeros_like(target) # Initialize weights to 0
        # TODO: not sure why we concat gru_estimated_flow with target, instead of directly init with target...
        # The first gru_estimated_flow is init to zero!
        self.gru_estimated_flow        = torch.cat([self.gru_estimated_flow, target], 1) ## Init gru flow with the one from reprojection!
        self.gru_estimated_flow_weight = torch.cat([self.gru_estimated_flow_weight, weight], 1)

    # Frees up memory as well,
    # TODO it doesn't do any marginalization?
    # TODO should we update kf_idx to re-use the empty slots?
    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """
        # store estimated factors that are removed
        if store:
            self.ii_inactive = torch.cat([self.ii_inactive, self.ii[mask]], 0)
            self.jj_inactive = torch.cat([self.jj_inactive, self.jj[mask]], 0)
            self.gru_estimated_flow_inactive = torch.cat([self.gru_estimated_flow_inactive, self.gru_estimated_flow[:,mask]], 1)
            self.gru_estimated_flow_weight_inactive = torch.cat([self.gru_estimated_flow_weight_inactive, self.gru_estimated_flow_weight[:,mask]], 1)

        # Actually remove edges
        self.ii  = self.ii[~mask]
        self.jj  = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == "volume":
            self.correlation_volumes = self.correlation_volumes[~mask]

        if self.gru_hidden_states is not None:
            self.gru_hidden_states = self.gru_hidden_states[:,~mask]

        if self.gru_contexts_input is not None:
            self.gru_contexts_input = self.gru_contexts_input[:,~mask]

        self.gru_estimated_flow        = self.gru_estimated_flow[:,~mask]
        self.gru_estimated_flow_weight = self.gru_estimated_flow_weight[:,~mask]

        # TODO: what about those keyframes that are isolated because no more factors touch them?

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inactive, self.jj_inactive)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def reproject(self, ii, jj, cam_T_body=None, jacobian=False):
        """ project points from ii -> jj """
        ii, jj = self.format_indicies(ii, jj, device=self.device) # TODO: is this really necessary?
        Gs = lietorch.SE3(self.cam0_T_world[None]) # TODO: this is a bit expensive no?

        # TODO: It would be great to visualize both
        coords, valid_mask, (Ji, Jj, Jz) = \
            pops.projective_transform(Gs, self.cam0_idepths[None], self.cam0_intrinsics[None], ii, jj, cam_T_body=cam_T_body, jacobian=jacobian)

        return coords, valid_mask, (Ji, Jj, Jz)

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.gru_estimated_flow_weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    @staticmethod
    def format_indicies(ii, jj, device="cpu"):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj


    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, images): # image must be normalized
        """ Context features """
        context_maps, gru_input_maps = self.context_net(images).split([128,128], dim=2)
        return context_maps.tanh().squeeze(0), gru_input_maps.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, images): # image must be normalized
        """ Features for correlation volume """
        return self.feature_net(images).squeeze(0)

    # images has shape [c, 3, h, w], where c is the number of images, h and w are the height and width of the images.
    # It outputs the images with an extra dimension at the front, the batch dimension b.
    # Further, it sends them to the GPU.
    def _normalize_imgs(self, images, droid_normalization=True):
        img_normalized = images[:,:,:3, ...] / 255.0 # Drop alpha channel
        if droid_normalization:
            mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
            stdv = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        else:
            mean = img_normalized.mean(dim=(3,4), keepdim=True)
            stdv = img_normalized.std(dim=(3,4), keepdim=True)
        img_normalized = img_normalized.sub_(mean).div_(stdv)
        return img_normalized


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def has_enough_motion(self, current_imgs_features):
        # Only calculates if enough motion by looking at cam0
        ic(self.last_kf_idx)
        last_img_features    = self.features_imgs[self.last_kf_idx][0]
        current_img_features = current_imgs_features[0]
        last_img_context     = self.contexts_imgs[self.last_kf_idx][0]
        last_img_gru_input   = self.cst_contexts_imgs[self.last_kf_idx][0]

        # Viz hidden features
        if self.viz:
            self.viz_hidden_features(last_img_features, current_img_features)

        # Index correlation volume
        corr = CorrBlock(last_img_features[None, None],
                         current_img_features[None, None])(self.coords0[None, None])  # TODO why not send the corr block?

        # Approximate flow magnitude using 1 update iteration
        _, delta, weight = self.update_net(last_img_context[None,None], last_img_gru_input[None,None], corr)

        # # Viz weight
        # if self.viz:
        #     self.viz_weight(weight)

        # # Viz flow
        # if self.viz:
        #     viz_flow(delta)#, flow_norm=100)

        # Check motion magnitude / add new frame to video
        has_enough_motion = delta.norm(dim=-1).mean().item() > self.motion_filter_thresh
        return has_enough_motion

    # Visulize the actual RGB image underneath!
    def viz_weight(self, weight, img_bg=None, ii=0, jj=0, write=True):
        weight = weight.permute(1,2,0)
        # Calculate the max norm of the weight image
        #weight_norm_max = weight.norm(dim=-1).max().item()
        #ic(weight_norm_max)
        #weight = weight / weight_norm_max
        weight_x = weight[...,0][None,None]
        weight_y = weight[...,1][None,None]
        # TODO should we multiply by 8 as in upflow8?
        new_size =(self.img_height, self.img_width)
        viz_weight_x = F.interpolate(weight_x, size=new_size, mode="bilinear", align_corners=True).squeeze()
        viz_weight_y = F.interpolate(weight_y, size=new_size, mode="bilinear", align_corners=True).squeeze()
        viz_weight_x = (viz_weight_x.to(torch.float).cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
        viz_weight_y = (viz_weight_y.to(torch.float).cpu().numpy()*255).astype(np.uint8) # scaling from 16bit to 8bit
        viz_weight_x = cv2.applyColorMap(viz_weight_x, cv2.COLORMAP_MAGMA)
        viz_weight_y = cv2.applyColorMap(viz_weight_y, cv2.COLORMAP_MAGMA)
        if img_bg is not None:
            img_bg = img_bg.permute(1,2,0).cpu().numpy()
            viz_weight_x = cv2.addWeighted(img_bg, 0.5, viz_weight_x, 0.5, 0)
            viz_weight_y = cv2.addWeighted(img_bg, 0.5, viz_weight_y, 0.5, 0)
        name_x = f'Weights X from {ii} to {jj}'
        name_y = f'Weights Y from {ii} to {jj}'
        cv2.imshow(name_x, viz_weight_x)
        cv2.imshow(name_y, viz_weight_y)
        if write:
            cv2.imwrite(name_x+".png", viz_weight_x)
            cv2.imwrite(name_y+".png", viz_weight_y)

    def viz_hidden_features(self, last_img_features, current_imgs_features):
        lif = last_img_features.norm(dim=0)
        cif = current_imgs_features.norm(dim=0)
        lif = lif / lif.max()
        cif = cif / cif.max()
        new_size =(self.img_height, self.img_width)
        lif = F.interpolate(lif[None,None], size=new_size, mode="bilinear", align_corners=True)
        cif = F.interpolate(cif[None,None], size=new_size, mode="bilinear", align_corners=True)
        lif = lif.squeeze()
        cif = cif.squeeze()
        cv2.imshow('Last Img Features', lif.cpu().numpy())
        cv2.imshow('Current Img Features', cif.cpu().numpy())

    def load_weights(self, weights_file):
        """ load trained model weights """

        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights_file).items()])
        state_dict = OrderedDict([
            (k.replace("fnet.", "feature_net."), v) for (k, v) in state_dict.items()])
        state_dict = OrderedDict([
            (k.replace("cnet.", "context_net."), v) for (k, v) in state_dict.items()])
        state_dict = OrderedDict([
            (k.replace("update.", "update_net."), v) for (k, v) in state_dict.items()])

        state_dict["update_net.weight.2.weight"] = state_dict["update_net.weight.2.weight"][:2]
        state_dict["update_net.weight.2.bias"]   = state_dict["update_net.weight.2.bias"][:2]
        state_dict["update_net.delta.2.weight"]  = state_dict["update_net.delta.2.weight"][:2]
        state_dict["update_net.delta.2.bias"]    = state_dict["update_net.delta.2.bias"][:2]

        return state_dict

    # TODO: I changed from kf0=1 to kf0=0
    def ba(self, gru_estimated_flow, gru_estimated_flow_weight, damping, ii, jj, 
           kf0=0, kf1=None, itrs=2, lm=1e-4, ep=0.1, 
           motion_only=False, compute_covariances=True):
        """ dense bundle adjustment (DBA): gets reduced camera matrix, solves it using gtsam, calculates covariances """

        if kf1 is None:
            # [kf0, kf1] window of bundle adjustment optimization
            kf1 = max(ii.max().item(), jj.max().item()) + 1

        N = kf1 - kf0
        HW = self.ht * self.wd
        kx = torch.unique(ii)

        kf_ids = [i+kf0 for i in range(kf1 - kf0)]
        f_ids = [self.kf_idx_to_f_idx[kf_id] for kf_id in kf_ids]

        Xii = np.array([X(f_id) for f_id in f_ids])

        initial_priors = None
        if f_ids[0] == 0:
            # Add strong prior
            if self.world_T_cam0_t0:
                _, initial_priors = self.get_gt_priors_and_values(kf_ids[0], f_ids[0])
            else:
                raise ("You need to add initial prior, or you'll have ill-cond hessian!")

        for _ in range(itrs):
            x0 = Values()
            linear_factor_graph = GaussianFactorGraph()

            for i in range(N):
                kf_id = i + kf0  
                x0.insert(Xii[i], gtsam.Pose3(SE3(self.world_T_body[kf_id]).matrix().cpu().numpy()))

            #ic(f"LINEARIZE iter: {iteration}")
            # At this point, cam0_poses and cam0_depths have been modified if ba() is called...
            # TODO: the saved self.* are to update the depths after GTSAM solve.
            # ic(target/weight.shape) ([42, 2, 44, 69])
            # Get H_vision, linearize at current Pose, Depth
            H, v, Q, E, w = droid_backends.reduced_camera_matrix(
                self.cam0_T_world,
                self.world_T_body, # TODO(remove, unnecessary, given the math)
                self.cam0_idepths,
                self.cam0_intrinsics[0],
                self.cam0_T_body,
                self.cam0_idepths_sensed,
                gru_estimated_flow, 
                gru_estimated_flow_weight, # TODO: we should pass as well previous pose covariances!, so to not lose information when optimizing
                damping,
                ii, jj, kf0, kf1)

            vision_factors = GaussianFactorGraph() 
            H = torch.nn.functional.unfold(H[None,None], (6, 6), stride=6).permute(2,0,1).view(N, N, 6, 6)
            v = torch.nn.functional.unfold(v[None,None], (6, 1), stride=6).permute(2,0,1).view(N, 6)
            H[range(N), range(N)] /= N # because we add N times the diagonal factors...
            v[:] /= N
            upper_triangular_indices = torch.triu_indices(N, N)
            # TODO: this is quite slow, we should implement it in c++/CUDA
            for i, j in zip(upper_triangular_indices[0], upper_triangular_indices[1]):
                if i == j:
                    vision_factors.add(HessianFactor(Xii[i], H[i, i].cpu().numpy(), v[i].cpu().numpy(), 0.0))
                else:
                    vision_factors.add(HessianFactor(Xii[i], Xii[j], H[i, i].cpu().numpy(), H[i, j].cpu().numpy(), v[i].cpu().numpy(), H[j, j].cpu().numpy(), v[j].cpu().numpy(), 0.0))
            linear_factor_graph.push_back(vision_factors)

            # Add Factors
            if initial_priors is not None:
                ic("ADDING initial prior!")
                linear_factor_graph.push_back(initial_priors.linearize(x0))

            # SOLVE
            # TODO: Droid does LM though, for which we should add to the diagonal
            # delta above has our delta, but I use GTSAM here because I don't want to convert delta to VectorValues...
            gtsam_delta = linear_factor_graph.optimizeDensely() # Calls Eigen Cholesky, without a particularly smart ordering (vs Eigen::SimplicialLLt...)
            self.last_state = x0.retract(gtsam_delta) # this retraction requires local right-hand deltas (chi)
            
            # RETRACT
            # Update world to Body Poses
            poses = gtsam.utilities.allPose3s(self.last_state)
            pose_keys = poses.keys()
            for i, key in enumerate(pose_keys):
                f_idx = gtsam.Symbol(key).index()
                kf_idx = self.f_idx_to_kf_idx[f_idx]
                self.world_T_body[kf_idx] = gtsam_pose_to_torch(poses.atPose3(key),
                                                                device=self.device, dtype=torch.float)
            
            # Update Cam to world Poses
            self.cam0_T_world[:] = (SE3(self.cam0_T_body[None]) * SE3(self.world_T_body).inv()).vec() # Refresh cam0_poses given retracted body_poses
            # Retract Depths
            xi_delta = torch.as_tensor(gtsam_delta.vector(pose_keys), device=self.device, dtype=torch.float).view(-1, 6)
            droid_backends.solve_depth(xi_delta, self.cam0_idepths, Q, E, w, ii, jj, kf0, kf1)
            self.cam0_idepths.clamp_(min=0.001)

        if compute_covariances:
            H, v = linear_factor_graph.hessian()
            L = None
            try:
                L = torch.linalg.cholesky(torch.as_tensor(H, device=self.device, dtype=torch.float))# from double to float...
            except Exception as e:
                print(e)
            if L is not None:
                identity = torch.eye(L.shape[0], device=L.device) # L has shape (PD,PD) 
                L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
                if torch.isnan(L_inv).any():
                    print("NANs in L_inv!!")
                    raise
                # We only care about block diagonals of sigma_g though, here we are calculating everything...
                sigma_gg = L_inv.transpose(-2,-1) @ L_inv 
                # TODO: this is the same as optimizeDensely in gtsam....
                # delta = sigma @ torch.as_tensor(v, device=self.device, dtype=torch.float)

                # Calculate sigmas
                # Extract only the block-diagonal of size D from sigma_g
                P = N
                D = L.shape[0] // P
                assert D == 6

                sigma_gg = sigma_gg.view(P, D, P, D).permute(0,2,1,3) # P x P x D x D
                sigma_g = torch.diagonal(sigma_gg, dim1=0, dim2=1).permute(2,0,1).view(P, D, D) # P x D x D

                Ei = E[:P]
                Ejz = E[P:P+ii.shape[0]]
                M = Ejz.shape[0]
                assert M == ii.shape[0]
                kx, kk = torch.unique(ii, return_inverse=True)
                K = kx.shape[0] # !!!! Aixo es different de N o rather de P i probablement K = P + fixed_poses, so K>P!

                # Ejz contains all the psi(D)*z(HW) pairs of products (M in total)
                # These are populating the off-diagonal of E,
                # The diagonal is populated by Ei
                min_ii_jj = min(ii.min(),jj.min())
                #ic(K)
                #ic(P)
                Ej = torch.zeros(K, K, D, HW, device=self.device) # HUGE MEMORY CONSUMPTION, this should be P, K, D, HW
                # The equation is E[jj[m], ii[m]] = Ejz[m] for m in M, but if we take into account the fixed poses, and indices, we get:
                Ej[jj - min_ii_jj, ii - min_ii_jj] = Ejz
                Ej = Ej[kf0-min_ii_jj:kf1-min_ii_jj].view(P,K,D,HW) # Keep only the keyframes we are optimizing over, and remove the fixed ones, but add all the depth-maps...
                # The diagonal is populated by Ei
                Ej[range(P), kf0-min_ii_jj:kf1-min_ii_jj, :, :] = Ei[range(P), :, :]
                
                E_sum = Ej
                E_sum = E_sum.view(P, K, D, HW)
                E_sum = E_sum.permute(0,2,1,3).reshape(P*D, K*HW)
                Q_ = Q.view(K*HW,1)
                F = torch.matmul(Q_ * E_sum.t(), L_inv) # K*HW x D*P
                F2 = torch.pow(F, 2)
                delta_cov = F2.sum(dim=-1) # K*HW
                z_cov = Q_.squeeze() + delta_cov # K*HW
                z_cov = z_cov.view(K, self.ht, self.wd)

                # Update body sigmas:
                for i, key in enumerate(pose_keys):
                    f_idx = gtsam.Symbol(key).index()
                    kf_idx = self.f_idx_to_kf_idx[f_idx]
                    self.world_T_body_cov[kf_idx] = sigma_g[i] # WARNING: this assumes that poses and sigma_g are in the same order!!
                # Update idepths_sigma, clamp?
                self.cam0_idepths_cov[kx] = z_cov
                # Update depths_sigma, clamp?
                depth_cov = z_cov / self.cam0_idepths[kx]**4
                self.cam0_depths_cov[kx] = depth_cov

        return x0, linear_factor_graph

    def get_gt_priors_and_values(self, kf_id, f_id):
        gt_pose = self.world_T_cam0_t0
        #gt_pose = np.eye(4)

        # Get prior
        pose_key = X(f_id)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])) # rot, trans
        pose_prior = PriorFactorPose3(pose_key, gtsam.Pose3(gt_pose), pose_noise)

        # Get guessed values
        x0 = Values()
        x0.insert(pose_key, gtsam.Pose3(gt_pose))

        # Add factors
        graph = NonlinearFactorGraph()
        graph.push_back(pose_prior)

        return x0, graph


    def backend(self, steps=12):
        """ main update """

        if not self.stereo and not torch.any(self.cam0_idepths_sensed):
            # TODO: we should normalize anyway if we have sensed idepths that are from the volume...
            #ic("Normalizing video.")
            self.normalize(self.kf_idx)

        self.max_factors = 16 * self.kf_idx
        self.corr_impl = "alt" 
        self.use_uncertainty = False

        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.age = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.correlation_volumes       = None
        self.gru_hidden_states         = None # initialized as context, but evolves as hidden state
        self.gru_contexts_input        = None # initialized as context, and remains as such
        self.damping = 1e-6 * torch.ones_like(self.cam0_idepths)

        self.gru_estimated_flow        = torch.zeros([1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)
        self.gru_estimated_flow_weight = torch.zeros([1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)

        # inactive factors
        self.ii_inactive = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_inactive = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.ii_bad  = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj_bad  = torch.as_tensor([], dtype=torch.long, device=self.device)

        self.gru_estimated_flow_inactive = torch.zeros([1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)
        self.gru_estimated_flow_weight_inactive = torch.zeros([1, 0, self.ht, self.wd, 2], device=self.device, dtype=torch.float)

        self.add_proximity_factors(rad=self.backend_radius,
                                   nms=self.backend_nms,
                                   thresh=self.backend_thresh,
                                   beta=self.beta) # uses self.max_factors to limit number of factors

        self.update_lowmem(steps=steps)
        self.clear_edges()
        self.viz_idx[:self.kf_idx] = True

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.gru_hidden_states         = None # initialized as context, but evolves as hidden state
        self.gru_contexts_input        = None # initialized as context, and remains as such

    def normalize(self, last_kf=-1):
        """ normalize depth and poses """
        s = self.cam0_idepths[:last_kf].mean() # mean inverse depth? whould it be the mean of the actual depth??
        self.cam0_idepths[:last_kf] /= s
        self.cam0_T_world[:last_kf, :3] *= s
        self.viz_idx[:last_kf] = True

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        #del self.frontend

        # TODO: maybe return between backend calls to see the updated geometry?
        if self.global_ba:
            torch.cuda.empty_cache()
            print("#" * 32)
            self.backend(7)

            torch.cuda.empty_cache()
            print("#" * 32)
            self.backend(12)

            torch.cuda.empty_cache()
            print("#" * 32)
            print("Clear memory")
            self.backend(0)
            torch.cuda.empty_cache()

            print("DONE WITH GLOBAL BA")
        else:
            print("Not running global BA...")
            torch.cuda.empty_cache()

        self.stop = True

    def get_viz_out(self, batch):
        viz_index, = torch.where(self.viz_idx)
        viz_out = None
        if len(viz_index) != 0:
            cam0_T_world       = torch.index_select(self.cam0_T_world, 0, viz_index)
            gt_poses           = torch.index_select(self.gt_poses, 0, viz_index)
            gt_depths          = torch.index_select(self.gt_depths, 0, viz_index)
            world_T_body       = torch.index_select(self.world_T_body, 0, viz_index)
            world_T_body_cov   = torch.index_select(self.world_T_body_cov, 0, viz_index)
            idepths            = torch.index_select(self.cam0_idepths, 0, viz_index)
            idepths_up         = torch.index_select(self.cam0_idepths_up, 0, viz_index)
            idepths_sensed     = torch.index_select(self.cam0_idepths_sensed, 0, viz_index)
            idepths_cov        = torch.index_select(self.cam0_idepths_cov, 0, viz_index)
            depths_cov         = torch.index_select(self.cam0_depths_cov, 0, viz_index)
            depths_cov_up      = torch.index_select(self.cam0_depths_cov_up, 0, viz_index)  # do not use up
            images             = torch.index_select(self.cam0_images, 0, viz_index)
            intrinsics         = torch.index_select(self.cam0_intrinsics, 0, viz_index) # are these the up or down intrinsics? (down!)

            if self.args.multi_gpu:
                # We cannot send viz_out in our device, if the receiving end is in another device.
                # Need to cpu-transer, which is super slow.
                out_device = "cpu"
            else:
                out_device = self.device

            ic(out_device)

            viz_out = {"cam0_poses":          cam0_T_world.to(device=out_device),
                       "gt_poses":            gt_poses.to(device=out_device),
                       "gt_depths":           gt_depths.to(device=out_device),
                       "world_T_body":        world_T_body.to(device=out_device),
                       "world_T_body_cov":    world_T_body_cov.to(device=out_device),
                       "cam0_idepths":        idepths.to(device=out_device),
                       "cam0_idepths_up":     idepths_up.to(device=out_device),
                       "cam0_idepths_sensed": idepths_sensed.to(device=out_device),
                       "cam0_idepths_cov":    idepths_cov.to(device=out_device),
                       "cam0_depths_cov":     depths_cov.to(device=out_device),
                       "cam0_depths_cov_up":  depths_cov_up.to(device=out_device),
                       "cam0_images":         images.to(device=out_device),
                       "cam0_intrinsics":     intrinsics.to(device=out_device), # intrinsics are downsampled
                       "calibs":              batch["calibs"], # calibs are the original ones
                       "viz_idx":             viz_index.to(device=out_device),
                       "kf_idx":              self.kf_idx,
                       "kf_idx_to_f_idx":     self.kf_idx_to_f_idx,
                       "is_last_frame":       batch["is_last_frame"]
                      }
            self.viz_idx[:] = False
        else:
            print("viz_index is empty, nothing to visualize")
            if batch["is_last_frame"]:
                ic("Last frame")
                # Leave all the entries null, this signals end of seq
                viz_out = {"is_last_frame": True}
            
        return viz_out
        