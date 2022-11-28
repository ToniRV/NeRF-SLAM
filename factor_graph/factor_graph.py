#!/usr/bin/env python3

from typing import Tuple

import torch as th
from torch import nn

from icecream import ic

import colored_glog as log

from factor_graph.variables import Variables

from gtsam import NonlinearFactorGraph as FactorGraph

class FactorGraphManager:
    def __init__(self) -> None:
        self.factor_graph = FactorGraph()

    def add(self, factor_graph):
        # Check we don't have repeated factors

        if factor_graph.empty():
            #log.warn("Attempted to add factors, but none were provided")
            return False

        # Remove None factors?

        # Add factors
        self.factor_graph.push_back(factor_graph)

        # Perhaps remove old factors

        # Update map from factor-id to slot
        return True

    def replace(self, key, factor):
        self.factor_graph.replace(key, factor)

    def remove(self, key):
        self.factor_graph.remove(key)

    def __iter__(self):
        return self.factor_graph.__iter__()

    def __getitem__(self, key):
        assert self.factor_graph.exists(key)
        return self.factor_graph.at(key)

    def __len__(self):
        # self.factor_graph.nrFactors()
        return self.factor_graph.size()

    def is_empty(self):
        return self.factor_graph.empty()

    def reset_factor_graph(self):
        self.factor_graph = FactorGraph()

    def get_factor_graph(self):
        return self.factor_graph
    

class TorchFactorGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.factors = nn.ModuleList([])
        ic(self.factors)
        self.run_jit = False

    def add(self, factors):
        # Check we don't have repeated factors

        # Append factors
        self.factors.extend(factors)

        # Update map from factor-id to slot
        pass

    def remove(self, factor_ids):
        pass

    def __iter__(self):
        return self.factors.__iter__()

    def __getitem__(self, key):
        return self.factors.__getitem__(key)

    def __len__(self):
        return len(self.factors)

    def is_empty(self):
        return self.__len__() == 0

    def forward(self, x: Variables) -> th.Tensor:
        return self._forward_jit(x) if self.run_jit else self._forward(x)

    def _forward(self, x: Variables) -> th.Tensor:
        residuals = th.stack([factor(x) for factor in self.factors])
        weights = th.stack([factor.weight(x) for factor in self.factors])
        return th.sum(th.pow(residuals, 2) * weights, dim=0)

    # Not necessarily faster...
    def _forward_jit(self, x: Variables) -> th.Tensor:
        residuals_calc = [th.jit.fork(factor, x) for factor in self.factors]
        weights_calc = [th.jit.fork(factor.weight, x) for factor in self.factors]
        residuals = th.stack([th.jit.wait(thread) for thread in residuals_calc])
        weights = th.stack([th.jit.wait(thread) for thread in weights_calc])
        return th.sum(th.pow(residuals, 2) * weights, dim=0)


    # TODO parallelize
    # These variables are already ordered.
    def linearize(self, x0: Variables) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Build the Jacobian matrix, here only for one f
        # Returns a tensor per each entry [i][j] corresponding to J_ij = d(output_i)/d(input_j)
        # this allows us to reason about the jacobian in terms of blocks.
        # A[i has dimensions of the output f(x0)][j has dimensions of the input x0]
        # vectorize=True --> first dimension is considered batch dimension
        # A = torch.autograd.functional.jacobian(f, x0, vectorize=True, create_graph=True)

        # Here we could query how to linearize:
        ## a) Linearize using AutoDiff (current)
        ## b) Linearize using closed-form jacobian
        ## extra) Linearize itself + compress with Schur complement, and return reduced Hessian matrix?
        AA = None; bb = None; ww = None
        # TODO Linearize factors in parallel...
        for factor in self.factors:
            if factor is None:
                log.warn("Got a None factor...")
                continue
            A, b, w = factor.linearize(x0)
            if AA is None:
                AA = A; bb = b; ww = w
                continue
            AA = th.hstack((AA, A))
            bb = th.hstack((bb, b))
            ww = th.hstack((ww, w))
        return AA, bb, ww

    # TODO parallelize
    def weight(self, x: Variables) -> th.Tensor:
        ww = None;
        for factor in self.factors:
            w = factor.weight(x)
            if ww is None:
                ww = w
                continue
            ww = th.hstack((ww, w))
        return ww
