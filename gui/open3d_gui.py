import numpy as np
from icecream import ic
import open3d as o3d
from fusion.tsdf_fusion import TsdfFusion

import numpy as np

import matplotlib as mpl # for colormaps

from utils.utils import *

from lietorch import SE3

import torch

import gtsam
from gtsam import Values

def to_o3d_device(device):
    if device == "cpu":
        return o3d.core.Device("CPU:0")
    elif "cuda" in device:
        idx = device[-1]
        return o3d.core.Device(f"CUDA:{idx}")
    else:
        raise NotImplementedError

class Open3dGui:
    OUT_PATH = None
    CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])
    CAM_LINES = np.array([
        [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

    # @render whatever function calls add_geometry or remove_geometry, this is a workaround to an o3d bug...
    def render(func):
        def _render(self, *args, **kwargs):
            self.viewport = self.viz.get_view_control().convert_to_pinhole_camera_parameters()

            # Drawing function, calls add_geometry etc...
            out = func(self, *args, **kwargs)

            if self.render_path and self.last_ix in self.history:
                cam = self.render_traj()
            else:
                cam = self.viewport

            if self.record_path:
                self.save_traj()

            # hack to allow interacting when using add_geometry
            self.viz.get_view_control().convert_from_pinhole_camera_parameters(cam)
            self.viz.poll_events()
            self.viz.update_renderer()
            return out
        return _render

    def render_traj(self):
        cam = o3d.camera.PinholeCameraParameters()
        cam0_pose = self.history[self.last_ix]["cam0_poses"]
        world_T_cam0    = SE3(cam0_pose).matrix().cpu().numpy()
        cam.intrinsic = self.viewport.intrinsic
        cam.extrinsic = world_T_cam0
        return cam

    def save_traj(self):
        import json
        if not self.OUT_PATH:
            self.OUT_PATH = "render_transforms.json"
            with open(self.OUT_PATH, "w") as file:
                json.dump({}, file)
        with open(self.OUT_PATH, "r") as file:
            data = json.load(file)
        data[int(self.last_ix)] =  self.viewport.extrinsic.tolist()
        with open(self.OUT_PATH, "w") as file:
            json.dump(data, file)

    def __init__(self, args, device, gt_pcl=None) -> None:
        self.device = device

        self.render_path = False
        self.record_path = True

        self.shutdown = False

        self.tsdf_fusion = None

        self.gt_pcl_on = False
        self.gt_pcl = gt_pcl

        self.output = None

        self.est_pcl_on = False

        self.colormap = mpl.colormaps['turbo']#.resampled(8)

        self.viz_world_frame = False
        self.viz_cams = True
        self.viz_body_frames = False
        self.viz_body_sigmas = True
        self.viz_o3d_pcl = True
        self.viz_gt_poses = False

        self.do_build_mesh = False
        self.do_eval_metrics = False
        self.do_rebuild_volume = False

        if args.mask_type == 'ours':
            self.depth_mask_type = "uncertainty"
        else:
            self.depth_mask_type = "uniform"

        self.scene = None # Raycasting scene

        self.debug = False

        self.mesh_count = 0
        self.redraw_est_pcl = False

        self.dsf = 8.0

        self.depth_scale = 1.0
        self.min_depth = 0.01
        self.max_depth = 6.0 # in m? but not up to scale...

        self.volumetric_fusion = False
        self.integrate_every_t  = 10 # a number of frames (>0), or None for no integration
        if self.volumetric_fusion:
            self.tsdf_fusion = TsdfFusion(None, args, None)

        self.do_render_volume = False

        self.evaluate = args.eval # Evaluate 2D metrics

        self.last_ix = 0

        # History of state
        self.history = {}

    # This is a workaround to avoid unpickable open3d objects
    def initialize(self):
        self.mesh = None
        self.write_3d_mesh = True
        self.o3d_intrinsics = None

        self.volume_file_name_count = 0

        self.o3d_device = to_o3d_device(self.device)

        #self.max_depth_sigma_thresh = 3.5 # in m? but not up to scale... Any depth with sigma higher than this is out of the recons or pcl/depth rendering
        self.max_depth_sigma_thresh = 10.0 # in m? but not up to scale... Any depth with sigma higher than this is out of the recons or pcl/depth rendering
        self.max_weight = 20.0 # This is in 1/variance units, so 1/m^2: variance median is (?) so use 1/(?)
        self.min_weight_for_render = 0.01 # render only the very accurate pixels
        self.min_weight_for_mesh = 0.1 # The smaller the more uncertain geometry that will be in the mesh.

        self.droid_camera_actors = {}
        self.body_actors = {}
        self.body_sigma_actors = {}
        self.gtsam_camera_actors = {}

        self.droid_pcl_actors = {}
        self.o3d_pcl_actors = {}

        self.viz = o3d.visualization.VisualizerWithKeyCallback()
        self.viz.create_window(width=640, height=480) #create_window(self, window_name='Open3D', width=1920, height=1080, left=50, top=50, visible=True)

        self.viz.get_render_option().point_size = 0.001 # in m
        self.viz.get_render_option().background_color = np.asarray([0, 0, 0]) # Black background
        self.viz.get_render_option().light_on = False

        self.viz.register_key_callback(ord("Q"), self.destroy_window)
        self.viz.register_key_callback(256, self.destroy_window) # esc
        self.viz.register_key_callback(ord("G"), self.toggle_gt_pcl)
        self.viz.register_key_callback(ord("E"), self.toggle_est_pcl)
        self.viz.register_key_callback(ord("K"), self.redraw_pcls)
        self.viz.register_key_callback(ord("M"), self.build_mesh)
        self.viz.register_key_callback(ord("N"), self.evaluate_metrics)
        self.viz.register_key_callback(ord("D"), self.write_mesh) # as in download...
        self.viz.register_key_callback(ord("S"), self.increase_filter)
        self.viz.register_key_callback(ord("A"), self.decrease_filter)
        self.viz.register_key_callback(ord("J"), self.increase_weight_filter)
        self.viz.register_key_callback(ord("H"), self.decrease_weight_filter) # Help button!
        self.viz.register_key_callback(ord("X"), self.remove_mesh)
        self.viz.register_key_callback(ord("T"), self.toggle_depth_mask_type)
        self.viz.register_key_callback(ord("Z"), self.rebuild_volume)

        if self.gt_pcl:
            self.gt_pcl = self.gt_pcl.create_pointcloud()

    # CREATORS
    def create_frame_actor(self, pose, scale=0.05):
        frame_actor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale,
                                                                        origin=np.array([0., 0., 0.]))
        frame_actor.transform(pose)
        return frame_actor

    def create_sphere_actor(self, pose, radius):
        # We create a line set because o3d doesn't have opacity yet...
        sphere_actor = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(radius=radius))
        color = (0.0, 0.0, 0.5)
        sphere_actor.paint_uniform_color(color)
        sphere_actor.transform(pose)
        return sphere_actor

    def create_camera_actor(self, color, pose, scale=0.05):
        camera_actor = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(scale * self.CAM_POINTS),
            lines=o3d.utility.Vector2iVector(self.CAM_LINES))
        camera_actor.paint_uniform_color(color)
        camera_actor.transform(pose)
        return camera_actor

    def create_pcl_actor(self, points, colors):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)
        return pcl

    # CALLBACKS

    def destroy_window(self, viz):
        if self.write_3d_mesh:
            self.write_mesh(viz)
        self.shutdown = True

    def toggle_gt_pcl(self, viz):
        if self.gt_pcl:
            if not self.gt_pcl_on:
                self.gt_pcl_on = True
                self.viz.add_geometry(self.gt_pcl, reset_bounding_box=True)
            else:
                self.gt_pcl_on = False
                self.viz.remove_geometry(self.gt_pcl, reset_bounding_box=True)
        else:
            print("Requested gt_pcl toggle but there is not gt_pcl stored...")

    def toggle_est_pcl(self, viz):
        pcl_actors = None
        if self.o3d_pcl_actors:
            pcl_actors = self.o3d_pcl_actors
        elif self.droid_pcl_actors:
            pcl_actors = self.droid_pcl_actors
        else:
            pcl_actors = None
        # Remove all pcl_actors
        if not self.est_pcl_on:
            self.est_pcl_on = True
            if pcl_actors is not None:
                for pcl in pcl_actors.values():
                    self.viz.add_geometry(pcl, reset_bounding_box=True)
        else:
            self.est_pcl_on = False
            if pcl_actors is not None:
                for pcl in pcl_actors.values():
                    self.viz.remove_geometry(pcl, reset_bounding_box=True)

    def remove_mesh(self, viz):
        print("Removing mesh.")
        if self.mesh:
            self.viz.remove_geometry(self.mesh)
        else:
            print("Requested mesh removal, but there is no mesh...")

    def toggle_depth_mask_type(self, viz):
        print("Toggle depth mask type.")
        if self.depth_mask_type == "uniform":
            print("Using Uncertainty mask type.")
            self.depth_mask_type = "uncertainty"
        else:
            print("Using Uniform mask type.")
            self.depth_mask_type = "uniform"

    def redraw_pcls(self,viz):
        # Calculate validity masks for the depth maps
        print(f"Redrawing pcls, this may take a while...")
        self.build_pcls(None)
        print("Done redrawing pcls.")

    def get_depth_masks(self, packet):
        cam0_depths_cov_up = packet["cam0_depths_cov_up"]
        masks = None
        if self.depth_mask_type == "uncertainty":
            masks = cam0_depths_cov_up.sqrt() < self.max_depth_sigma_thresh
        elif self.depth_mask_type == "uniform":
            masks = torch.ones_like(cam0_depths_cov_up).to(torch.bool)
        else:
            print(f"ERROR: Unknown depth mask type: {self.depth_mask_type}")
        return masks

    @render # To render on every new pcl
    def draw_pcl(self, ix, rgbd, intrinsics, cam_pose):
        if ix in self.o3d_pcl_actors:
            self.viz.remove_geometry(self.o3d_pcl_actors[ix])
            del self.o3d_pcl_actors[ix]

        pcl_actor = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics, project_valid_depth_only=True)
        pcl_actor.transform(cam_pose)

        self.o3d_pcl_actors[ix] = pcl_actor
        if self.est_pcl_on:
            self.viz.add_geometry(pcl_actor, reset_bounding_box=False)

    def build_mesh(self, viz=None):
        ic("Requesting mesh...")
        self.do_build_mesh = True

    def evaluate_metrics(self, viz=None):
        # Eval by rendering from all viewpoints.
        ic("Requesting evaluation...")
        self.do_eval_metrics = True

    def write_mesh(self, viz):
        if self.mesh:
            mesh_file_name = f"Mesh{self.mesh_count}.ply"
            print(f"Writing mesh to file: {mesh_file_name}")
            o3d.io.write_triangle_mesh(mesh_file_name, self.mesh)
            self.mesh_count +=1
        else:
            print("Requested write_mesh, but you need to first build the mesh!")

    def increase_filter(self, viz):
        self.max_depth_sigma_thresh += 0.1 
        print(f"Increasing Depth Sigma filter_tresh: {self.max_depth_sigma_thresh}")
        self.redraw_est_pcl = True

    def decrease_filter(self, viz):
        self.max_depth_sigma_thresh -= 0.1 if self.max_depth_sigma_thresh >= 0.0 else 0.0
        print(f"Decreasing Depth Sigma filter_tresh: {self.max_depth_sigma_thresh}")
        self.redraw_est_pcl = True

    def increase_weight_filter(self, viz):
        self.min_weight_for_mesh += 0.01 # mesh it all
        print(f"Increasing mesh weight filter : {self.min_weight_for_mesh}")
        if self.min_weight_for_mesh >= self.max_weight:
            self.min_weight_for_mesh = self.max_weight
            print(f"ERROR: cannot increase min_weight_for_mesh above max_weight: {self.max_weight}")

    def decrease_weight_filter(self, viz):
        self.min_weight_for_mesh -= 0.01 # mesh it all
        print(f"Decreasing mesh weight filter : {self.min_weight_for_mesh}")
        if self.min_weight_for_mesh <= 0.0:
            self.min_weight_for_mesh = 0.0
            print("ERROR: cannot decrease min_weight_for_mesh below 0.0")

    def rebuild_volume(self, viz):
        print("Requesting to rebuild volume")
        self.do_rebuild_volume = True

    # Main LOOP
    def visualize(self, data_packets):
        if self.shutdown:
            self.viz.destroy_window()
            return False

        self.add_geometry(data_packets)

        # Send output for downstream tasks (fusion)
        tmp_output = {"depth_mask_type": self.depth_mask_type,
                      "build_mesh": {"min_weight_for_mesh": self.min_weight_for_mesh} if self.do_build_mesh else False,
                      "rebuild_volume": self.do_rebuild_volume,
                      "eval_metrics": self.do_eval_metrics}

        self.do_build_mesh = False
        self.do_rebuild_volume = False
        self.do_eval_metrics = False
        
        if self.output == tmp_output:
            # Same output as before, just skip sending an output...
            return None
        else:
            self.output = tmp_output
            return tmp_output

    @render
    def add_geometry(self, data_packets):
        if self.viz_world_frame:
            self.viz.add_geometry(self.create_frame_actor(np.eye(4), scale=0.5), reset_bounding_box=True)
            self.viz_world_frame = False

        if data_packets: # data_packets is a dict of data_packets
            for name, packet in data_packets.items():
                if name == "data":
                    self.viz_data_packet(packet)
                elif name == "slam":
                    self.viz_slam_packet(packet)
                elif name == "fusion":
                    self.viz_fusion_packet(packet)
                else:
                    print(f"viz_{name}_packet not implemented...")
                    raise NotImplementedError

    def viz_data_packet(self, packet):
        print("Visualizing DATA packet.")

        if type(packet) is Values:
            values = packet
            cam0_poses = gtsam.utilities.allPose3s(values)
            keys = gtsam.KeyVector(cam0_poses.keys())
            if len(keys) > 0:
                for key in keys:
                    if values.exists(key):
                        if key in self.gtsam_camera_actors:
                            self.viz.remove_geometry(self.gtsam_camera_actors[key])
                            del self.gtsam_camera_actors[key]

                        gtsam_world_T_body = values.atPose3(key)
                        gtsam_world_T_cam0 = gtsam_world_T_body * self.body_T_cam0
                        color = (0.5, 0.25, 0.45)
                        camera_actor = self.create_camera_actor(
                            color, gtsam_world_T_cam0.matrix(), scale=0.05)
                        self.gtsam_camera_actors[key] = camera_actor
                        self.viz.add_geometry(camera_actor, reset_bounding_box=True)

        if self.viz_gt_poses:
            gt_df = packet["gt_t0_t1"]
            if not gt_df.empty:
                nearest_t_cam0 = gt_df.index.get_indexer([packet["t_cams"][0]], method="nearest")[0]
                world_T_body = get_pose_from_df(gt_df.iloc[nearest_t_cam0])
                world_T_cam0 = world_T_body @ packet["cam_calibs"][0].body_T_cam
                world_T_cam1 = world_T_body @ packet["cam_calibs"][1].body_T_cam
                #self.viz.add_geometry(self.create_frame_actor(world_T_body), reset_bounding_box=True)
                color = (0.0, 0.5, 0.9)
                self.viz.add_geometry(self.create_camera_actor(color, world_T_cam0, scale=0.01), reset_bounding_box=True)
                #self.viz.add_geometry(self.create_camera_actor(1, world_T_cam1), reset_bounding_box=True)
                
    def viz_slam_packet(self, packet):
        if not packet:
            print("Missing o3d input packet from SLAM module...")
            return True

        packet = packet[1]
        if packet is None:
            print("O3d packet from SLAM module is None...")
            return True

        print("Visualizing SLAM packet.")
        if self.evaluate and packet["is_last_frame"] and not "cam0_images" in packet:
            print("Last Frame reached, and no global BA")
            self.evaluate_metrics()
            return True

        # Store/Update history of packets
        if not self.update_history(packet):
            print("End of SLAM sequence, nothing more to visualize...")
            return True

        if self.o3d_intrinsics is None:
            cam0_images     = packet["cam0_images"]
            cam0_intrinsics = packet["cam0_intrinsics"]
            _, h, w = cam0_images[0].shape
            fx, fy, cx, cy = np.asarray(self.dsf * cam0_intrinsics[0].cpu().numpy())
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        if self.redraw_est_pcl:
            self.build_pcls(None)
            self.redraw_est_pcl = False
        elif self.viz_o3d_pcl:
            self.build_pcls(packet)

        if self.viz_cams:
            self.vis_cams(packet)

        if self.viz_body_frames:
            self.vis_body_frames(packet)

        if self.viz_body_sigmas:
            self.vis_body_sigmas(packet)

        #if self.mesh_every_t and self.kf_idx % self.mesh_every_t == 0:
        #    self.build_mesh() # TODO: needs the volume

        if self.volumetric_fusion and self.tsdf_fusion:
            if packet is None:
                print("Rebuilding pcl from history.")
                packet = self.get_history_packet()
                print("Done rebuilding pcl from history.")
            # TODO: perhaps only integrate the first pcls, not the last which is not well optimized yet?
            #if self.integrate_every_t and self.kf_idx % self.integrate_every_t == 0:
            #    print(f"Integrate RGBD {self.kf_idx} in volume...")
            if self.depth_mask_type == "uniform": packet["cam0_depths_cov_up"] = torch.ones_like(packet["cam0_depths_cov_up"])
            self.tsdf_fusion.build_volume(packet, self.o3d_intrinsics, self.get_depth_masks(packet))

        if self.do_render_volume and self.tsdf_fusion:
            self.tsdf_fusion.render_volume(packet, self.o3d_intrinsics,
                                           self.min_weight_for_render,
                                           self.max_depth_sigma_thresh)

        if self.evaluate and packet["is_last_frame"]:
            print("Last Frame reached...")
            self.evaluate_metrics()

    def viz_fusion_packet(self, packet):
        if not packet:
            print("Missing Gui input packet from Fusion module...")
            return True

        if packet is None:
            print("Gui packet from Fusion module is None...")
            return True

        print("Visualizing Gui packet from Fusion")
        mesh = packet["mesh"]
        if mesh:
            self.viz_mesh(mesh.create_mesh())
        else:
            print("Did not receive mesh...")

    def viz_mesh(self, mesh):
        if self.mesh:
            self.viz.remove_geometry(self.mesh)
        self.viz.add_geometry(mesh, reset_bounding_box=True)
        self.mesh = mesh

    def update_history(self, packet):
        if packet["is_last_frame"]:
            return False
        kf_idx                 = packet["kf_idx"]
        viz_idx                = packet["viz_idx"]
        cam0_poses             = packet["cam0_poses"]
        cam0_depths_cov_up     = packet["cam0_depths_cov_up"]
        world_T_body           = packet["world_T_body"]
        world_T_body_cov       = packet["world_T_body_cov"]
        cam0_idepths_up        = packet["cam0_idepths_up"]
        cam0_images            = packet["cam0_images"]
        cam0_intrinsics        = packet["cam0_intrinsics"]
        kf_idx_to_f_idx        = packet["kf_idx_to_f_idx"]
        calibs                 = packet["calibs"]
        gt_depths              = packet["gt_depths"]
        for i, ix in enumerate(viz_idx): # zip
            ix = kf_idx_to_f_idx[ix.item()]
            if not ix in self.history.keys():
                self.history[ix] = {}
            h = self.history[ix]
            h["kf_idx"]                 = kf_idx
            h["viz_idx"]                = viz_idx[i] # no need to store it since ix == viz_idx[i]
            h["cam0_poses"]             = cam0_poses[i]
            h["cam0_depths_cov_up"]     = cam0_depths_cov_up[i]
            h["world_T_body"]           = world_T_body[i]
            h["world_T_body_cov"]       = world_T_body_cov[i]
            h["cam0_idepths_up"]        = cam0_idepths_up[i]
            h["cam0_images"]            = cam0_images[i]
            h["cam0_intrinsics"]        = cam0_intrinsics[i]
            h["calibs"]                 = calibs[0]
            h["gt_depths"]              = gt_depths[i]
        self.last_ix = kf_idx_to_f_idx[viz_idx[-1].item()]
        return True

    def vis_cams(self, packet):
        viz_idx                = packet["viz_idx"]
        cam0_poses             = packet["cam0_poses"]
        world_T_cam0           = SE3(cam0_poses).inv().matrix().cpu().numpy()
        for i in range(len(viz_idx)):
            ix = viz_idx[i].item()
            if ix in self.droid_camera_actors:
                self.viz.remove_geometry(self.droid_camera_actors[ix])
                del self.droid_camera_actors[ix]

            cam_pose     = world_T_cam0[i]
            color = (0.5, 0.25, 0.45)
            camera_actor = self.create_camera_actor(color, cam_pose, scale=0.01)

            self.droid_camera_actors[ix] = camera_actor
            self.viz.add_geometry(camera_actor, reset_bounding_box=True)

    def vis_body_frames(self, packet):
        viz_idx                = packet["viz_idx"]
        world_T_body           = SE3(packet["world_T_body"].matrix().cpu().numpy())
        for i in range(len(viz_idx)):
            ix = viz_idx[i].item()
            if ix in self.body_actors:
                self.viz.remove_geometry(self.body_actors[ix])
                del self.body_actors[ix]

            body_actor = self.create_frame_actor(world_T_body[i])

            self.body_actors[ix] = body_actor
            self.viz.add_geometry(body_actor, reset_bounding_box=True)
            
    def vis_body_sigmas(self, packet):
        viz_idx                = packet["viz_idx"]
        world_T_body           = SE3(packet["world_T_body"]).matrix().cpu().numpy()
        world_T_body_cov       = packet["world_T_body_cov"]
        for i in range(len(viz_idx)):
            ix = viz_idx[i].item()
            if ix in self.body_sigma_actors:
                self.viz.remove_geometry(self.body_sigma_actors[ix])
                del self.body_sigma_actors[ix]

            body_pose = world_T_body[i]
            body_cov = world_T_body_cov[i]
            translation_cov = body_cov[3:,3:] # last 3x3 block is translation (gtsam convention)
            _, S, _ = torch.linalg.svd(translation_cov)
            # TODO: O3D does not support ellipsoids yet...
            body_sigma_actor = self.create_sphere_actor(body_pose, radius=S.max().sqrt().item())

            s = S.max().sqrt().item()
            ic(s)
            max_s = 0.3
            s_norm =  s / max_s if s < max_s else 1.0
            color = self.colormap(s_norm)[:3]
            body_sigma_actor = self.create_camera_actor(color, body_pose, scale=0.02)

            self.body_sigma_actors[ix] = body_sigma_actor
            self.viz.add_geometry(body_sigma_actor, reset_bounding_box=True)

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
            packet["gt_depths"]          += [h["gt_depths"]]
        packet["viz_idx"]            = torch.stack(packet["viz_idx"]           )
        packet["cam0_poses"]         = torch.stack(packet["cam0_poses"]        )
        packet["cam0_depths_cov_up"] = torch.stack(packet["cam0_depths_cov_up"])
        packet["cam0_idepths_up"]    = torch.stack(packet["cam0_idepths_up"]   )
        packet["cam0_images"]        = torch.stack(packet["cam0_images"]       )
        packet["cam0_intrinsics"]    = torch.stack(packet["cam0_intrinsics"]   )
        #packet["calibs"]             = packet["calibs"]
        packet["gt_depths"]          = torch.stack(packet["gt_depths"]         )
        return packet

    def build_pcls(self, packet):
        if packet is None:
            print("Rebuilding pcl from history.")
            packet = self.get_history_packet()
            print("Done rebuilding pcl from history.")

        viz_idx         = packet["viz_idx"]
        cam0_poses      = packet["cam0_poses"]
        cam0_depths     = packet["cam0_idepths_up"].pow(-1)
        cam0_images     = packet["cam0_images"].permute(0,2,3,1)
        world_T_cam0    = SE3(cam0_poses).inv().matrix().cpu().numpy()

        # Add last pcl
        masks = self.get_depth_masks(packet)
        if masks is not None:
            cam0_depths[~masks] = self.max_depth + 1.0 # The ~mask values will be truncated when building rgbd frame, and the same in the volume.

        cam0_depths = cam0_depths.contiguous()
        cam0_images = cam0_images.contiguous()
        for ix, cam_pose, depth, color in zip(viz_idx, world_T_cam0, cam0_depths, cam0_images):
            # Using dlpack should be much faster, avoids copying, but I think the to_legacy() below might just be copying...
            o3d_depth  = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depth)))
            o3d_color  = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(color)))
            #o3d_depth = o3d.t.geometry.Image(o3d.core.Tensor(depth, device=self.o3d_device))
            #o3d_color = o3d.t.geometry.Image(o3d.core.Tensor(color, device=self.o3d_device))

            # Generate Pointcloud (for visualization only)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color.to_legacy(),
                                                                      o3d_depth.to_legacy(),
                                                                      depth_scale=self.depth_scale,
                                                                      depth_trunc=self.max_depth,
                                                                      convert_rgb_to_intensity=False)

            self.draw_pcl(ix.item(), rgbd, self.o3d_intrinsics, cam_pose)