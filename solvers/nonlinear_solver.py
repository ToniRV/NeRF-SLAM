#!/usr/bin/env python3

from abc import abstractclassmethod

import colored_glog as log

import torch as th

from factor_graph.variables import Variable, Variables
from factor_graph.factor_graph import TorchFactorGraph

from solvers.linear_solver import LinearLS, LinearSolverType

import gtsam
from gtsam import (ISAM2, LevenbergMarquardtOptimizer, NonlinearFactorGraph, PriorFactorPose2, Values)
from gtsam import NonlinearFactorGraph as FactorGraph

from icecream import ic

class Solver:
    def __init__(self):
        pass

    @abstractclassmethod
    def solve(self, factor_graph, x0):
        raise

class iSAM2(Solver):

    def __init__(self):
        super().__init__()
        # Set ISAM2 parameters and create ISAM2 solver object
        isam_params = gtsam.ISAM2Params()

        # Dogleg params
        dogleg_params = gtsam.ISAM2DoglegParams()
        dogleg_params.setInitialDelta(1.0)
        dogleg_params.setWildfireThreshold(1e-5) 
        dogleg_params.setVerbose(True)
        # dogleg_params.setAdaptationMode(string adaptationMode);

        # Gauss-Newton params
        gauss_newton_params = gtsam.ISAM2GaussNewtonParams()
        gauss_newton_params.setWildfireThreshold(1e-5)

        # Optimization parameters
        isam_params.setOptimizationParams(gauss_newton_params)
        isam_params.setFactorization("CHOLESKY") # QR or Cholesky

        # Linearization parameters
        isam_params.enableRelinearization = True
        isam_params.enablePartialRelinearizationCheck = False
        isam_params.setRelinearizeThreshold(0.1) # TODO
        isam_params.relinearizeSkip = 10

        # Memory efficiency, but slower
        isam_params.findUnusedFactorSlots = True

        # Debugging parameters, disable for speed
        isam_params.evaluateNonlinearError = True
        isam_params.enableDetailedResults = True

        #isam_params.print()

        self.isam2 = gtsam.ISAM2(isam_params)

    def solve(self, factor_graph, x0):
        # factors_to_remove = factor_graph.keyVector() # since all are structureless, and we re-add all inertial
        # print(factors_to_remove)
        # self.isam2.update(factor_graph, x0, factors_to_remove)
        self.isam2.update(factor_graph, x0) # Only one iteration!!
        result = self.isam2.calculateEstimate()
        delta = self.isam2.getDelta()
        return result, delta

# class GaussNewton(Solver):
#     def __init__(self):
#         super().__init__()
#         self.params = gtsam.GaussNewtonParams()
#         self.params.setVerbosityLM("SUMMARY")
#         self.x0 = Values()
# 
#     def solve(self, factor_graph, x0):
#         self.x0.insert(x0)
#         optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, self.x0, self.params)
#         return optimizer.optimize()

class LevenbergMarquardt(Solver):
    def __init__(self):
        super().__init__()
        self.params = gtsam.LevenbergMarquardtParams()
        # static void SetLegacyDefaults(LevenbergMarquardtParams* p) {
        # // Relevant NonlinearOptimizerParams:
        # p->maxIterations = 100;
        # p->relativeErrorTol = 1e-5;
        # p->absoluteErrorTol = 1e-5;
        # // LM-specific:
        # p->lambdaInitial = 1e-5;
        # p->lambdaFactor = 10.0;
        # p->lambdaUpperBound = 1e5;
        # p->lambdaLowerBound = 0.0;
        # p->minModelFidelity = 1e-3;
        # p->diagonalDamping = false;
        # p->useFixedLambdaFactor = true;
        self.params.setVerbosityLM("SUMMARY")
        self.x0 = Values()

    def solve(self, factor_graph, x0):
        self.x0.insert(x0)
        optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, self.x0, self.params)
        return optimizer.optimize()
    

class NonlinearLS(Solver):
    def __init__(self, iterations=2, min_x_update=0.001, linear_solver=LinearSolverType.Cholesky):
        super().__init__()
        self.iters = iterations
        self.min_x_update = min_x_update
        if linear_solver == LinearSolverType.Inverse:
            self.linear_solver = LinearLS.solve_inverse
        elif linear_solver == LinearSolverType.Cholesky:
            self.linear_solver = LinearLS.solve_cholesky
        else:
            raise ValueError(f'Unknown linear solver: {linear_solver}')

        # Set all data members to 0
        self._reset_history()

        # Logging
        self.log_every_n = 1

    # f is a nonlinear function of the states resulting in measurements
    # w is the measurement weights
    # x0 is the linearization point (initial guess)
    def solve(self, fg: TorchFactorGraph, x0: Variables) -> th.Tensor:
        # Start iterative call to nonlinear solve
        self._reset_history()
        return self._solve_nonlinear(fg, x0, 1)

    # TODO(Toni): abstract this, and perhaps even call iSAM2 from gtsam...
    # f is expected to be in standard form, meaning weighted/whitened.
    def _solve_nonlinear(self, fg: TorchFactorGraph, x0: Variables, i: int) -> th.Tensor:
        if fg.is_empty():
            log.warn("Factor graph is empty, returning initial guess...")
            return x0

        # 1. Linearize nonlinear system with respect to x, this can do Schur complement as well.
        A, b, w = fg.linearize(x0)

        # 2. Solve linear system
        delta_x = self.linear_solver(A, b, w)

        if delta_x is None:
            log.warn("Linear system is not invertible, returning initial guess...")
            return x0
       
        # 3. Retract/Update
        x_new = Variables()
        for var in x0.retract(delta_x).items():
            x_new.add(var[1])

        # 4. Store best solutions so far
        with th.no_grad():
            loss_new = fg(x_new)
            if self.x_best is None or th.sum(loss_new) < th.sum(self.loss_best):
                self.x_best = x_new
                self.loss_best = loss_new

            # Logging
            self._logging(x0, delta_x, loss_new, i)

        # 5. Repeat until we reach termination condition
        return x_new if self.terminate(i, x_new) else self._solve_nonlinear(fg, x_new, i + 1)

    def _logging(self, x0, delta_x, loss, i):
            self.x0_list.append(x0)
            self.delta_x_list.append(delta_x)

            if i % self.log_every_n == 0:
                # Print sum over batches loss
                print(f"inner_loss: {th.sum(loss):>7f}  [{i:>5d}/{self.iters:>5d}]")


    # should be abstract
    def terminate(self, i, x):
        # one implementation is just to look at how many iters we have done
        convergence = False
        # with th.no_grad():
        #     ic(x.shape)
        #     # We need to determine per-batch convergence :O
        #     # And how do we avoid per-batch updates for the converged ones?
        #     convergence = th.abs(x - self.x0_list[-1]) < self.x_tol
        #     convergence = self.delta_x_list[-1] < self.min_x_update
        return i >= self.iters or convergence 

            
    def _reset_history(self):
        self.x_best = None
        self.loss_best = None
        self.A_list = []
        self.b_list = []
        self.x0_list = []
        self.delta_x_list = []
