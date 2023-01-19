import numpy as np

from icecream import ic
import open3d as o3d

from lietorch import SE3

from utils.flow_viz import viz_depth_map, viz_depth_sigma
from utils.utils import *
from utils.open3d_pickle import _MeshTransmissionFormat

import cv2
import torch

def to_o3d_device(device):
    if device == "cpu":
        return o3d.core.Device("CPU:0")
    elif "cuda" in device:
        idx = device[-1]
        return o3d.core.Device(f"CUDA:{idx}")
    else:
        raise NotImplementedError

class TsdfFusion:
    def __init__(self, name, args, device) -> None:
        self.device = device

        self.debug = False

        self.dsf = 8.0 # down-scaling factor, hardcoded in frontend

        self.depth_scale = 1.0
        self.min_depth = 0.01
        self.max_depth = 6.0 # in m? but not up to scale...

        self.evaluate = args.eval # Evaluate 2D metrics

        # History of state
        self.history = {}

        self.o3d_intrinsics = None
        self.max_depth_sigma_thresh = 10000 # in m? but not up to scale... Any depth with sigma higher than this is out of the recons or pcl/depth rendering
        self.min_weight_for_render = 0.01 # render only the very accurate pixels

        self.volume_file_name_count = 0

        self.do_render_volume = False

        if name == "sigma":
            self.depth_mask_type = "uncertainty"
        else:
            self.depth_mask_type = "uniform"

        self.initialize()

    # This is a workaround to avoid unpickable open3d objects
    def initialize(self):
        self.volume_file_name_count = 0

        self.o3d_device = to_o3d_device(self.device)

        # TODO: make thread-safe?
        self.max_weight = 20.0 # This is in 1/variance units, so 1/m^2: variance median is (?) so use 1/(?)

        self.voxel_size = 6.0 / 512 # representing a 3m x 3m x 3m (m = meter) room with a dense 512 x 512 x 512 voxel grid
        self.block_resolution = 16
        self.block_count = 5000
        self.voxel_length = 0.010  # in m
        self.sdf_trunc = 0.10  # in m
        self.use_old_volume = False

        self.volume = None
        if self.use_old_volume:
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_length,
                sdf_trunc=self.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            self.volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight', 'color'),
                attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=self.voxel_size, 
                block_resolution=self.block_resolution,
                block_count=self.block_count,
                device=self.o3d_device)

    # Main LOOP
    def fuse(self, data_packets):
        gui_output = None
        if data_packets: # data_packets is a dict of data_packets
            for name, packet in data_packets.items():
                if name == "slam":
                    self.handle_slam_packet(packet)
                elif name == "gui":
                    gui_output = self.handle_gui_packet(packet)
                else:
                    raise NotImplementedError("Unrecognized input packet for TsdfFusion Module")
        if gui_output:
            return gui_output
        else:
            return None
        #return {"data": data_output, "slam": slam_output, "gui": gui_output} # return None if we want to shutdown
    
    def handle_slam_packet(self, packet):
        #print("Received SLAM packet for TsdfFusion")
        if not packet:
            print("Missing Fusion input packet from SLAM module...")
            return True

        packet = packet[1]
        if packet is None:
            print("Fusion packet from SLAM module is None...")
            return True

        if self.evaluate and packet["is_last_frame"] and not "cam0_images" in packet:
            print("Last Frame reached, and no global BA")
            if self.history:
                self.evaluate_metrics()
            else:
                print("Can't evaluate volume since there is no history!")
            return True

        # Store/Update history of packets
        if not self.update_history(packet):
            print("End of SLAM sequence, nothing more to fuse...")
            return True

        if self.o3d_intrinsics is None:
            cam0_images     = packet["cam0_images"]
            cam0_intrinsics = packet["cam0_intrinsics"]
            _, h, w = cam0_images[0].shape
            fx, fy, cx, cy = np.asarray(self.dsf * cam0_intrinsics[0].cpu().numpy())
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # TODO: perhaps only integrate the first pcls, not the last which is not well optimized yet?
        #if self.integrate_every_t and self.kf_idx % self.integrate_every_t == 0:
        #    print(f"Integrate RGBD {self.kf_idx} in volume...")
        if self.depth_mask_type == "uniform": packet["cam0_depths_cov_up"] = torch.ones_like(packet["cam0_depths_cov_up"])
        self.build_volume(packet, self.o3d_intrinsics, self.get_depth_masks(packet))

        if self.do_render_volume:
            self.render_volume(packet, self.o3d_intrinsics, self.min_weight_for_render, self.max_depth_sigma_thresh)

        if self.evaluate and packet["is_last_frame"]:
            print("Last Frame reached...")
            self.evaluate_metrics()

    def handle_gui_packet(self, packet):
        print("Received GUI packet for TsdfFusion")

        output = None
        if not packet:
            print(f"Received gui packet in tsdf_fusion, but it is {packet}")
            return output

        self.depth_mask_type = packet["depth_mask_type"]
        if packet["build_mesh"]:
            mesh = self.build_mesh(packet["build_mesh"]["min_weight_for_mesh"])
            if mesh:
                output = {"mesh" : _MeshTransmissionFormat(mesh)}
            else:
                print("Could not build a mesh...")
        if packet["rebuild_volume"]:
            self.rebuild_volume()
        if packet["eval_metrics"]:
            self.evaluate_metrics()

        return output
            
    def evaluate_metrics(self):
        # Eval by rendering from all viewpoints.
        print("Evaluating Reconstruction.")
        print("\t Rebuilding pcl from history.")
        packet = self.get_history_packet()
        print("\t Done rebuilding pcl from history.")
        print("\t Rendering and evaluating reconstruction.")
        self.render_volume(packet, self.o3d_intrinsics,
                           self.min_weight_for_render,
                           self.max_depth_sigma_thresh,
                           evaluate=True)
        print("Done evaluating reconstruction.")

    # builds volume from packet # TODO: build volume from pcl directly.
    def build_volume(self, packet, o3d_intrinsics, masks):
        poses             = packet["cam0_poses"]
        depths            = packet["cam0_idepths_up"].to(device=self.device).pow(-1)
        depths_weights    = packet["cam0_depths_cov_up"].to(device=self.device).pow(-1).sqrt() # the more std deviation, the less weight.
        images            = packet["cam0_images"].to(device=self.device).permute(0,2,3,1).contiguous() # / 255.0 # bcs volume integrate needs so, but rgbd?
        cam0_T_world      = SE3(poses).matrix().cpu().numpy()

        if masks is not None:
            # The ~mask values will be truncated when building rgbd frame, and the same in the volume.
            depths[~masks] = self.max_depth + 1.0 

        intrinsic      = np.ascontiguousarray(o3d_intrinsics.intrinsic_matrix)
        depths         = depths.contiguous()
        depths_weights = depths_weights.contiguous()
        images         = images.contiguous()
        for cam_pose, depth, depth_weight, color in zip(cam0_T_world, depths, depths_weights, images):
            color_float = color.float()

            # Using dlpack should be much faster, avoids copying, but I think the to_legacy() below might just be copying...
            o3d_depth         = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depth)))
            o3d_depth_weight  = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depth_weight)))
            o3d_color         = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(color)))
            o3d_color_float   = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(color_float)))

            extrinsic = o3d.core.Tensor(cam_pose.astype(np.float64))

            # Fuse volumetrically
            if self.use_old_volume:
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color.to_legacy(),
                                                                          o3d_depth.to_legacy(),
                                                                          depth_scale=self.depth_scale,
                                                                          depth_trunc=self.max_depth,
                                                                          convert_rgb_to_intensity=False)
                self.volume.integrate(rgbd, intrinsic, extrinsic)
            else:
                self.custom_volume_integrate(o3d_depth, o3d_depth_weight, o3d_color_float, intrinsic, extrinsic)

    def rebuild_volume(self):
        print("Rebuilding pcl from history.")
        packet = self.get_history_packet()
        print("Done rebuilding pcl from history.")
        self.reset_volume()
        print("Integrating all pointclouds into the volume... Might take a while...")
        if self.depth_mask_type == "uniform": packet["cam0_depths_cov_up"] = torch.ones_like(packet["cam0_depths_cov_up"])
        self.build_volume(packet, self.o3d_intrinsics, self.get_depth_masks(packet))
        print("Done integrating pointclouds into the volume.")

    def custom_volume_integrate(self, o3d_depth, o3d_depth_weights, o3d_color_float, intrinsic, extrinsic):
        # THIS is also computed for the ray-cast, maybe combine?
        frustum_block_coords = None
        try:
            frustum_block_coords = self.volume.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, self.depth_scale, self.max_depth)
        except Exception as e:
            print(e)
            ic(self.max_depth)
            return False

        self.volume.hashmap().activate(frustum_block_coords)
        buf_indices, vol_masks = self.volume.hashmap().find(frustum_block_coords)
        o3d.core.cuda.synchronize()

        voxel_coords, voxel_indices = self.volume.voxel_coordinates_and_flattened_indices(buf_indices)
        o3d.core.cuda.synchronize()

        extrinsic_dev    = extrinsic.to(self.o3d_device)
        extrinsic_dev    = extrinsic_dev.to(o3d.core.float64)
        voxel_coords_dev = voxel_coords.to(o3d.core.float64)

        xyz = extrinsic_dev[:3, :3] @ voxel_coords_dev.T() + extrinsic_dev[:3, 3:]
        intrinsic_dev = o3d.core.Tensor(intrinsic).to(self.o3d_device, o3d.core.float64)
        uvd = intrinsic_dev @ xyz
        d = uvd[2]
        u = (uvd[0] / d).round().to(o3d.core.int64)
        v = (uvd[1] / d).round().to(o3d.core.int64)
        o3d.core.cuda.synchronize()

        mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < o3d_depth.columns) & (v < o3d_depth.rows)
        v_proj = v[mask_proj]
        u_proj = u[mask_proj]
        d_proj = d[mask_proj]
        depth_readings = o3d_depth.as_tensor()[v_proj, u_proj, 0].to(o3d.core.float32) / self.depth_scale
        sdf = depth_readings - d_proj.to(o3d.core.float32)

        mask_inlier = (depth_readings > 0) & (depth_readings < self.max_depth) & (sdf >= -self.sdf_trunc) #& (depth_sigma_readings < self.max_depth_sigma_thresh)

        sdf[sdf >= self.sdf_trunc] = self.sdf_trunc # saturate sdf
        sdf = sdf / self.sdf_trunc # normalize sdf
        o3d.core.cuda.synchronize()

        weight = self.volume.attribute('weight').reshape((-1, 1))
        tsdf   = self.volume.attribute('tsdf').reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]

        # Increase Weights
        weight_readings = o3d_depth_weights.as_tensor()[v_proj, u_proj, 0].to(o3d.core.float32)
        w = weight[valid_voxel_indices]
        wr = weight_readings[mask_inlier].reshape(w.shape)
        wp = w + wr

        # Update TSDF
        tsdf[valid_voxel_indices] = (
            w * tsdf[valid_voxel_indices] + wr * sdf[mask_inlier].reshape(w.shape)) / (wp)

        # Update Color
        color_readings = o3d_color_float.as_tensor()[v_proj, u_proj, :3].to(o3d.core.float32)
        color_volume = self.volume.attribute('color').reshape((-1, 3))
        # TODO: account for normal view point...
        color_volume[valid_voxel_indices] = (
            w * color_volume[valid_voxel_indices] + wr * color_readings[mask_inlier]) / (wp)

        # Update weight
        wp[wp > self.max_weight] = self.max_weight  # saturate the weights
        weight[valid_voxel_indices] = wp
        o3d.core.cuda.synchronize()

        return True

    def reset_volume(self):
        ic("Resetting volume")
        if self.volume and self.use_old_volume:
            self.volume.reset()
        else:
            # Re-init?
            self.volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight', 'color'),
                attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                attr_channels=((1), (1), (3)), # representing a 3m x 3m x 3m (m = meter) room with a dense 512 x 512 x 512 voxel grid
                voxel_size=self.voxel_size / 2,
                block_resolution=self.block_resolution,
                block_count=self.block_count,
                device=self.o3d_device)

    def render_volume(self, packet, o3d_intrinsics, min_weight_for_render, max_depth_sigma_thresh, evaluate=False):
        if not packet:
            print("Error: asked to render volume, but packet is None.")
            return

        viz_idx                = packet["viz_idx"]
        cam0_poses             = packet["cam0_poses"]
        cam0_depths_cov_up     = packet["cam0_depths_cov_up"]
        cam0_idepths_up        = packet["cam0_idepths_up"]
        cam0_images            = packet["cam0_images"]
        gt_depths              = packet["gt_depths"]
        world_T_cam0           = SE3(cam0_poses).inv().matrix().cpu().numpy()
        H, W = o3d_intrinsics.height, o3d_intrinsics.width

        if evaluate:
            import pandas
            data_frame = pandas.DataFrame(columns=['Dt','PSNR', 'L1'])

        # Raycast the scene and get us dense depth maps:
        # Create the scene
        if self.use_old_volume:
            if self.mesh and self.mesh.has_triangles():
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
                self.scene = o3d.t.geometry.RaycastingScene() # Do we need to reinit all the time?
                _ = self.scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
            else:
                print("No mesh available, using empty scene...")
        else:
            if not self.volume:
                print("No TSDF volume available, first build the volume...")
                return

        total_psnr = 0
        total_l1 = 0
        for i in range(len(viz_idx)):
            ix          = viz_idx[i]
            cam_pose    = world_T_cam0[i]
            color       = cam0_images[i].permute(1,2,0)# / 255.0 # just for debug/viz
            depth       = cam0_idepths_up[i].pow(-1)
            depth_sigma = cam0_depths_cov_up[i].sqrt()
            gt_depth    = gt_depths[i].squeeze()

            # TODO: render at the low resolution!
            # Build the rays
            intrinsic = np.ascontiguousarray(o3d_intrinsics.intrinsic_matrix)
            extrinsic = o3d.core.Tensor(np.linalg.inv(cam_pose).astype(np.float64))
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=intrinsic,
                extrinsic_matrix=extrinsic,
                width_px=W,
                height_px=H)

            print(f"Casting Rays for camera {ix}")
            render = None
            if self.use_old_volume:
                if self.scene:
                    render = self.scene.cast_rays(rays)
                else:
                    continue
            else:
                frustum_block_coords = None
                try:
                    # o3d_depth just needs to be a rough depth estimate.... But why can't we just render at a viewpoint?
                    o3d_depth = o3d.t.geometry.Image(o3d.core.Tensor(depth.cpu().numpy(), device=self.o3d_device))
                    frustum_block_coords = self.volume.compute_unique_block_coordinates(
                        o3d_depth, intrinsic, extrinsic, self.depth_scale, self.max_depth)
                except Exception as e:
                    print(e)
                    continue

                render = self.volume.ray_cast(block_coords=frustum_block_coords,
                                              intrinsic=intrinsic,
                                              extrinsic=extrinsic,
                                              width=W,
                                              height=H,
                                              render_attributes=[
                                                  'depth', 'color', 'index', 'interp_ratio'],
                                              depth_scale=self.depth_scale,
                                              depth_min=self.min_depth,
                                              depth_max=self.max_depth,
                                              weight_threshold=min_weight_for_render,
                                              range_map_down_factor=16)

            # Render depth directly
            rendered_depth = o3d.t.geometry.Image(render['depth'])

            # Render color directly
            rendered_c = o3d.t.geometry.Image(render['color'])

            # Render via indexing
            nb_indices = render['index'].reshape((-1))
            nb_interp_ratio = render['interp_ratio'].reshape((-1, 1))

            # Render color
            rendered_color = self.volume.attribute('color').reshape((-1, 3))
            nb_colors = rendered_color[nb_indices] * nb_interp_ratio
            sum_colors = nb_colors.reshape((H, W, 8, 3)).sum((2)) / 255.0

            # Render weights
            rendered_weights = self.volume.attribute('weight').reshape((-1, 1))
            nb_weights = rendered_weights[nb_indices] * nb_interp_ratio
            sum_weights = nb_weights.reshape((H, W, 8, 1)).sum((2)) # interpolated weights

            # Fuse rendered depth map with the current depth map estimate on the non-masked values
            rendered_depth_map = None
            if self.use_old_volume:
                rendered_depth_map = torch.tensor(
                    render['t_hit'].numpy(), device=depth.device, dtype=depth.dtype)
            else:
                rendered_depth_map = torch.tensor(
                    render['depth'].cpu().numpy(), device=depth.device, dtype=depth.dtype).squeeze()

            if self.debug:
                cv2.imshow("Rendered depth", rendered_depth.colorize_depth(
                    self.depth_scale, self.min_depth, self.max_depth).as_tensor().cpu().numpy())
                cv2.imshow("Rendered color?", rendered_c.as_tensor().cpu().numpy())
                cv2.imshow('rendered color', sum_colors.cpu().numpy())
                cv2.imshow('actual color', color.cpu().numpy())
                cv2.imshow('rendered weights', sum_weights.cpu().numpy())
                if self.use_old_volume:
                    cv2.imshow("rend_depth_map", render['t_hit'].numpy())
                    rendered_depth_mask = (np.abs(render['geometry_ids'].numpy()) > 0).astype(float)
                    cv2.imshow('Rendered Depth Mask',rendered_depth_mask)

                viz_depth_map(rendered_depth_map, fix_range=False, name="rendered_depth_map")
                viz_depth_map(depth, fix_range=False, name="estimated_depth_map")
                viz_depth_map(gt_depth, fix_range=False, name="ground-truth depth")
                viz_depth_sigma(depth_sigma[None], name="estimated_depth_map sigma",
                                fix_range=True,
                                bg_img=color.permute(2,0,1)[None],
                                sigma_thresh=max_depth_sigma_thresh)
                scale = gt_depth.mean() / rendered_depth_map.mean()
                diff_depth_map = torch.abs(scale * rendered_depth_map - gt_depth)
                viz_depth_map(diff_depth_map, fix_range=False, name="Error depth map", colormap=cv2.COLORMAP_TURBO, invert=False)
                cv2.waitKey(1)


            # Calc metrics
            if evaluate:
                # Calculate PSNR 
                rendered_c = rendered_c.as_tensor()
                color = color / 255.0
                mse = float(compute_error(rendered_c.cpu().numpy(), color.cpu().numpy()))
                psnr = mse2psnr(mse)
                total_psnr += psnr

                # Calculate L1 for depth: rendered depth vs ground-truth depth
                rendered_depth_map = rendered_depth_map[:-8,:-8] # Remove the last row of the image which is not rendered correctly bcs of an O3D bug
                gt_depth = gt_depth[:-8,:-8]
                scale = gt_depth.mean() / rendered_depth_map.mean()
                diff_depth_map = torch.abs(scale * rendered_depth_map - gt_depth)
                diff_depth_map[diff_depth_map > 2.0] = 2.0 # Truncate outliers to 1m, otw biases metric, this can happen either bcs depth is not estimated or bcs gt depth is wrong. 
                if self.debug:
                    viz_depth_map(diff_depth_map, fix_range=False, name="Error depth map after crop", colormap=cv2.COLORMAP_TURBO, invert=False)
                    cv2.waitKey(1)
                l1 = diff_depth_map.mean().cpu().numpy() * 100 # From m to cm AND use the mean (as in Nice-SLAM)
                total_l1 += l1

        if evaluate:
            #dt = self.ngp.elapsed_training_time
            dt = 0 # TODO: calc time, is it needed? not really...
            psnr = total_psnr / (len(viz_idx) or 1)
            l1 = total_l1 / (len(viz_idx) or 1)
            print(f"Dt={dt}; PSNR={psnr}; L1={l1}")
            data_frame.loc[len(data_frame.index)] = [dt, psnr, l1]
            data_frame.to_csv("results.csv")

    def update_history(self, packet):
        if packet["is_last_frame"]:
            return False
        kf_idx                 = packet["kf_idx"]
        viz_idx                = packet["viz_idx"]
        cam0_poses             = packet["cam0_poses"]
        cam0_depths_cov_up     = packet["cam0_depths_cov_up"]
        cam0_idepths_up        = packet["cam0_idepths_up"]
        cam0_images            = packet["cam0_images"]
        cam0_intrinsics        = packet["cam0_intrinsics"]
        gt_depths              = packet["gt_depths"]
        calibs                 = packet["calibs"]
        kf_idx_to_f_idx        = packet["kf_idx_to_f_idx"]
        for i, ix in enumerate(viz_idx): 
            ix = kf_idx_to_f_idx[ix.item()]
            if not ix in self.history.keys():
                self.history[ix] = {}
            h = self.history[ix]
            h["kf_idx"]                 = kf_idx
            h["viz_idx"]                = viz_idx[i] # no need to store it since ix == viz_idx[i]
            h["cam0_poses"]             = cam0_poses[i]
            h["cam0_depths_cov_up"]     = cam0_depths_cov_up[i]
            h["cam0_idepths_up"]        = cam0_idepths_up[i]
            h["cam0_images"]            = cam0_images[i]
            h["cam0_intrinsics"]        = cam0_intrinsics[i]
            h["gt_depths"]              = gt_depths[i]
            h["calibs"]                 = calibs[0]
        return True

    def get_history_packet(self):
        packet = {}
        packet["viz_idx"]            = []
        packet["cam0_poses"]         = []
        packet["cam0_depths_cov_up"] = []
        packet["cam0_idepths_up"]    = []
        packet["cam0_images"]        = []
        packet["cam0_intrinsics"]    = []
        packet["calibs"]             = []
        packet["gt_depths"]          = []
        for h in self.history.values():
            packet["viz_idx"]            += [h["viz_idx"]]
            packet["cam0_poses"]         += [h["cam0_poses"]]
            packet["cam0_depths_cov_up"] += [h["cam0_depths_cov_up"]]
            packet["cam0_idepths_up"]    += [h["cam0_idepths_up"]]
            packet["cam0_images"]        += [h["cam0_images"]]
            packet["cam0_intrinsics"]    += [h["cam0_intrinsics"]]
            packet["calibs"]             += [h["calibs"]]
            # TODO: WHY ARE WE DIVIDING? SHOULDN'T IT BE MULTIPLYING?!? 
            packet["gt_depths"]          += [torch.tensor(h["gt_depths"], device=h["cam0_images"].device, dtype=torch.float32).permute(2,0,1) * h["calibs"].depth_scale]
        packet["viz_idx"]            = torch.stack(packet["viz_idx"]           )
        packet["cam0_poses"]         = torch.stack(packet["cam0_poses"]        )
        packet["cam0_depths_cov_up"] = torch.stack(packet["cam0_depths_cov_up"])
        packet["cam0_idepths_up"]    = torch.stack(packet["cam0_idepths_up"]   )
        packet["cam0_images"]        = torch.stack(packet["cam0_images"]       )
        packet["cam0_intrinsics"]    = torch.stack(packet["cam0_intrinsics"]   )
        #packet["calibs"]             = packet["calibs"]
        packet["gt_depths"]          = torch.stack(packet["gt_depths"]         )
        return packet

    def get_depth_masks(self, packet):
        cam0_depths_cov_up     = packet["cam0_depths_cov_up"]
        masks = None
        if self.depth_mask_type == "uncertainty":
            masks = cam0_depths_cov_up.sqrt() < self.max_depth_sigma_thresh
        elif self.depth_mask_type == "uniform":
            masks = torch.ones_like(cam0_depths_cov_up).to(torch.bool)
        else:
            raise NotImplementedError(f"Unknown depth mask type: {self.depth_mask_type}")
        return masks

    def build_mesh(self, min_weight_for_mesh):
        mesh = None
        if self.volume:
            if self.use_old_volume:
                mesh = self.volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
            else:
                try:
                    mesh = self.volume.extract_triangle_mesh(weight_threshold=min_weight_for_mesh).to_legacy()
                    mesh.compute_vertex_normals()
                except Exception as e:
                    print(e)
                    print("Failed to extract mesh on GPU, saving for offline meshing...")
                    self.volume.save(f"volume{self.volume_file_name_count}.npz")
                    self.volume_file_name_count += 1
        else:
            print("No available volume to mesh...")
        return mesh

    def stop_condition(self):
        return False