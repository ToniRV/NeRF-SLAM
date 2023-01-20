#!/usr/bin/env python3


from __future__ import print_function

from abc import abstractclassmethod

from icecream import ic

import torch as th
from torch import nn
from factor_graph.variables import Variable, Variables
from factor_graph.factor import Factor
from factor_graph.factor_graph import TorchFactorGraph

from slam.meta_slam import SLAM
import numpy as np

import gtsam
from gtsam import (ImuFactor, Pose3, Rot3, Point3)
from gtsam import PriorFactorPose3, PriorFactorConstantBias, PriorFactorVector
from gtsam.symbol_shorthand import B, V, X


def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)


class InertialFrontend(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def forward(self, mini_batch):
        pass
    

import gtsam
from gtsam import Values
from gtsam import NonlinearFactorGraph as FactorGraph
from gtsam.symbol_shorthand import B

import torch as th


class PreIntegrationInertialFrontend(InertialFrontend):
    def __init__(self):
        super().__init__()
        self.last_key = None

        # THESE ARE A LOT OF ALLOCATIONS: #frames * imu_buffer_size * 7  * 2 (because double precision!)
        # self.imu_buffer = 200
        # self.imu_t0_t1 = th.empty(self.imu_buffer, 7, device='cpu', dtype=th.float64)#.share_memory_() 

    def initialize_imu_frontend():
        pass

    def forward(self, mini_batch, last_state):
        # Call IMU preintegration from kimera/gtsam or re-implement...
        # The output of PreIntegrationInertialFrontend is a
        # bunch of (pose,vel,bias)-to-(pose,vel,bias) factors.
        print("PreIntegrationInertialFrontend.forward")
        k = mini_batch["k"]
        if last_state is None and k == 0:
            self.last_key = k
            # Initialize the preintegration
            self.imu_params = mini_batch["imu_calib"]
            self.preint_imu_params = self.get_preint_imu_params(self.imu_params)
            initial_state = self.initial_state()
            self.pim = gtsam.PreintegratedImuMeasurements(self.preint_imu_params,
                                                          initial_state[2])
            # Set initial priors and values
            x0, factors = self.initial_priors_and_values(k, initial_state)
            return x0, factors

        imu_t0_t1 = mini_batch["imu_t0_t1"]
        # imu_meas_count = len(imu_t0_t1_df)
        # self.imu_t0_t1[:imu_meas_count] = th.as_tensor(imu_t0_t1_df, device='cpu', dtype=th.float64).cpu().numpy()

        # Integrate measurements between frames (101 measurements)
        #self.preintegrate_imu(self.imu_t0_t1, imu_meas_count)
        self.preintegrate_imu(imu_t0_t1, -1)
        #self.pim.print()

        #if k % 10 != 0: # simulate keyframe selection
        #    return Values(), FactorGraph()

        # Get factors
        imu_factor = ImuFactor(X(self.last_key), V(self.last_key), X(k), V(k), B(self.last_key), self.pim)
        bias_btw_factor = self.get_bias_btw_factor(k, self.pim.deltaTij())

        # Add factors to inertial graph
        graph = FactorGraph() 
        graph.add(imu_factor)
        graph.add(bias_btw_factor)

        print("LAST STATE")
        print(last_state)
        # Get guessed values (from IMU integration)
        last_W_Pose_B, last_W_Vel_B, last_imu_bias = \
            last_state.atPose3(X(self.last_key)), \
            last_state.atVector(V(self.last_key)),\
            last_state.atConstantBias(B(self.last_key))
        last_navstate = gtsam.NavState(last_W_Pose_B, last_W_Vel_B)
        new_navstate = self.pim.predict(last_navstate, last_imu_bias);

        x0 = Values()
        x0.insert(X(k), new_navstate.pose())
        x0.insert(V(k), new_navstate.velocity())
        x0.insert(B(k), last_imu_bias)

        self.last_key = k
        self.pim.resetIntegrationAndSetBias(last_imu_bias)

        return x0, graph

    # k: current key, n: number of imu measurements
    def preintegrate_imu(self, imu_t0_t1, n):
        meas_acc = imu_t0_t1[:n, 4:7]
        meas_gyr = imu_t0_t1[:n, 1:4]
        delta_t = (imu_t0_t1[1:n, 0] - imu_t0_t1[0:n-1, 0]) * 1e-9 # convert to seconds
        for acc, gyr, dt in zip(meas_acc, meas_gyr, delta_t):
            self.pim.integrateMeasurement(acc, gyr, dt)
        # TODO: fix this loop!
        #self.pim.integrateMeasurements(meas_acc, meas_gyr, delta_t)

    def get_bias_btw_factor(self, k, delta_t_ij):
        # Bias evolution as given in the IMU metadata
        sqrt_delta_t_ij = np.sqrt(delta_t_ij);
        bias_sigmas_acc = sqrt_delta_t_ij * self.imu_params.a_b * np.ones(3)
        bias_sigmas_gyr = sqrt_delta_t_ij * self.imu_params.g_b * np.ones(3)
        bias_sigmas = np.concatenate((bias_sigmas_acc, bias_sigmas_gyr), axis=None)
        bias_noise_model = gtsam.noiseModel.Diagonal.Sigmas(bias_sigmas)
        bias_value = gtsam.imuBias.ConstantBias()
        return gtsam.BetweenFactorConstantBias(B(self.last_key), B(k), bias_value, bias_noise_model) 

    # Send IMU priors
    def initial_priors_and_values(self, k, initial_state):
        pose_key = X(k)
        vel_key = V(k)
        bias_key = B(k)

        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01]))
        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.001)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)

        initial_pose, initial_vel, initial_bias = initial_state

        # Get inertial factors 
        pose_prior = PriorFactorPose3(pose_key, initial_pose, pose_noise)
        vel_prior = PriorFactorVector(vel_key, initial_vel, vel_noise)
        bias_prior = PriorFactorConstantBias(bias_key, initial_bias, bias_noise)

        # Add factors to inertial graph
        graph = FactorGraph()
        graph.push_back(pose_prior)
        graph.push_back(vel_prior)
        graph.push_back(bias_prior)

        # Get guessed values
        x0 = Values()
        x0.insert(pose_key, initial_pose)
        x0.insert(vel_key, initial_vel)
        x0.insert(bias_key, initial_bias)

        return x0, graph

    def initial_state(self):
        true_pose = gtsam.Pose3(gtsam.Rot3(0.060514,-0.828459,-0.058956,-0.553641), # qw, qx, qy, qz
                                               gtsam.Point3(0.878612,2.142470,0.947262))
        true_vel = np.array([0.009474,-0.014009,-0.002145])
        true_bias = gtsam.imuBias.ConstantBias(np.array([-0.012492,0.547666,0.069073]), np.array([-0.002229,0.020700,0.076350]))
        naive_pose = gtsam.Pose3() #identity
        naive_vel = np.zeros(3)
        naive_bias = gtsam.imuBias.ConstantBias()
        initial_pose = true_pose
        initial_vel = true_vel
        initial_bias = true_bias
        return initial_pose, initial_vel, initial_bias

    def get_preint_imu_params(self, imu_calib):
        I = np.eye(3)
        preint_params = gtsam.PreintegrationParams(imu_calib.n_gravity);
        preint_params.setAccelerometerCovariance(np.power(imu_calib.a_n, 2.0) * I)
        preint_params.setGyroscopeCovariance(np.power(imu_calib.g_n, 2.0) * I)
        preint_params.setIntegrationCovariance(np.power(imu_calib.imu_integration_sigma, 2.0) * I)
        preint_params.setUse2ndOrderCoriolis(False)
        preint_params.setOmegaCoriolis(np.zeros(3, dtype=float))
        preint_params.print()
        return preint_params