#!/usr/bin/env python3

import numpy as np
import open3d as o3d
from icecream import ic

class MeshRenderer:
    def __init__(self, mesh_path, intrinsics, resolution) -> None:
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        # Do we need to register the gt_mesh to the SLAM frame of ref? I don't think so because we start SLAM with gt pose
        # Nonethelesssss, we need to register the SLAM's estimated trajectory with the ground-truth using Sim(3)!
        # And scale the gt_mesh with the resulting scale parameter,
        # OR! Just render depths at the ground-truth trajectory and register the depth-maps with scale? (I think that is what ORbeez doess)

        focal_length = intrinsics[:2]
        principal_point = intrinsics[2:]
        fx, fy = focal_length[0], focal_length[1]
        cx, cy = principal_point[0], principal_point[1]
        w, h = resolution[0], resolution[1]

        self.cam = o3d.camera.PinholeCameraParameters()
        self.cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=w, height=h)
        self.viz.get_render_option().mesh_show_back_face = True

        self.mesh_uploaded = False

        self.viz_world_frame = False

    def render_mesh(self, c2w):
        viewport = self.viz.get_view_control().convert_to_pinhole_camera_parameters()

        self.cam.extrinsic = np.linalg.inv(c2w)

        if self.viz_world_frame:
            self.viz.add_geometry(self.create_frame_actor(np.eye(4), scale=0.5), reset_bounding_box=True)
            self.viz_world_frame = False

        if not self.mesh_uploaded:
            self.viz.add_geometry(self.mesh, reset_bounding_box=True)
            self.mesh_uploaded = True

        #ctr = self.viz.get_view_control()
        #ctr.set_constant_z_far(20)
        #ctr.convert_from_pinhole_camera_parameters(self.cam)

        self.viz.poll_events()
        self.viz.update_renderer()

        gt_depth = self.viz.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)

        # hack to allow interacting when using add_geometry
        viewport = self.viz.get_view_control().convert_from_pinhole_camera_parameters(viewport)

        self.viz.poll_events()
        self.viz.update_renderer()

        return gt_depth


    def create_frame_actor(self, pose, scale=0.05):
        frame_actor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale,
                                                                        origin=np.array([0., 0., 0.]))
        frame_actor.transform(pose)
        return frame_actor