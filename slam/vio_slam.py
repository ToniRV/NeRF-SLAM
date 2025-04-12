#!/usr/bin/env python3

from abc import abstractclassmethod

from icecream import ic

from torch import nn
from factor_graph.variables import Variable, Variables
from factor_graph.factor_graph import TorchFactorGraph

from gtsam import Values
from gtsam import NonlinearFactorGraph
from gtsam import GaussianFactorGraph

from slam.meta_slam import SLAM
from slam.inertial_frontends.inertial_frontend import PreIntegrationInertialFrontend
from slam.visual_frontends.visual_frontend import RaftVisualFrontend
from solvers.nonlinear_solver import iSAM2, LevenbergMarquardt, Solver

########################### REMOVE ############################
import numpy as np
import gtsam
from gtsam import (ImuFactor, Pose3, Rot3, Point3)
from gtsam import PriorFactorPose3, PriorFactorConstantBias, PriorFactorVector
from gtsam.symbol_shorthand import B, V, X


# Send IMU priors
def initial_priors_and_values(k, initial_state):
    pose_key = X(k)
    vel_key = V(k)
    bias_key = B(k)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.000001, 0.000001, 0.000001, 0.00001, 0.00001, 0.00001]))
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.000001)
    bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.00001)

    initial_pose, initial_vel, initial_bias = initial_state

    # Get inertial factors 
    pose_prior = PriorFactorPose3(pose_key, initial_pose, pose_noise)
    vel_prior = PriorFactorVector(vel_key, initial_vel, vel_noise)
    bias_prior = PriorFactorConstantBias(bias_key, initial_bias, bias_noise)

    # Add factors to inertial graph
    graph = NonlinearFactorGraph()
    graph.push_back(pose_prior)
    graph.push_back(vel_prior)
    graph.push_back(bias_prior)

    # Get guessed values
    x0 = Values()
    x0.insert(pose_key, initial_pose)
    x0.insert(vel_key, initial_vel)
    x0.insert(bias_key, initial_bias)

    return x0, graph

def initial_state():
    true_world_T_imu_t0 = gtsam.Pose3(gtsam.Rot3(0.060514, -0.828459, -0.058956, -0.553641),  # qw, qx, qy, qz
                                      gtsam.Point3(0.878612, 2.142470, 0.947262))
    true_vel = np.array([0.009474,-0.014009,-0.002145])
    true_bias = gtsam.imuBias.ConstantBias(np.array([-0.012492,0.547666,0.069073]), np.array([-0.002229,0.020700,0.076350]))
    naive_pose = gtsam.Pose3.Identity()
    naive_vel = np.zeros(3)
    naive_bias = gtsam.imuBias.ConstantBias()
    initial_pose = true_world_T_imu_t0
    initial_vel = true_vel
    initial_bias = true_bias
    initial_pose = naive_pose
    initial_vel = naive_vel
    initial_bias = naive_bias
    return initial_pose, initial_vel, initial_bias
###############################################################


class VioSLAM(SLAM):
    def __init__(self, name, args, device):
        super().__init__(name, args, device)
        world_T_imu_t0, _, _ = initial_state()
        imu_T_cam0 = Pose3(np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                               [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                               [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                               [0.0, 0.0, 0.0, 1.0]]))

        imu_T_cam0 = Pose3(np.eye(4))

        #world_T_imu_t0 = Pose3(args.world_T_imu_t0)
        #world_T_imu_t0 = Pose3(np.eye(4))
        world_T_imu_t0 = Pose3(np.array(
            [[-7.6942980e-02, -3.1037781e-01,  9.4749427e-01,  8.9643948e-02],
             [-2.8366595e-10, -9.5031142e-01, -3.1130061e-01,  4.1829333e-01],
             [ 9.9703550e-01, -2.3952398e-02,  7.3119797e-02,  4.8306200e-01],
             [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]))

        self.visual_frontend = RaftVisualFrontend(world_T_imu_t0, imu_T_cam0, args, device=device)
        #self.inertial_frontend = PreIntegrationInertialFrontend()

        self.backend = iSAM2()

        self.values = Values()
        self.inertial_factors = NonlinearFactorGraph()
        self.visual_factors = NonlinearFactorGraph()

        self.last_state = None
    
    def stop_condition(self):
        return self.visual_frontend.stop_condition()

    # Converts sensory inputs to measurements and initial guess
    def _frontend(self, batch, last_state, last_delta):
        # Compute optical flow
        x0_visual, visual_factors, viz_out = self.visual_frontend(batch)  # TODO: currently also calls BA, and global BA
        self.last_state = x0_visual

        if x0_visual is None:
            return False

        # Wrap guesses
        x0 = Values()
        factors = NonlinearFactorGraph()

        return x0, factors, viz_out

    def _backend(self, factor_graph, x0):
        return self.backend.solve(factor_graph, x0)
    
