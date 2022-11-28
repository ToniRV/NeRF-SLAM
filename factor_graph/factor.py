#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List

import torch as th
from torch import nn

from factor_graph.loss_function import LossFunction
from factor_graph.key import Key
from factor_graph.variables import Variable, Variables

class Factor(ABC, nn.Module):
    def __init__(self, keys: List[Key], loss_function: LossFunction, device='cpu') -> None:
        super().__init__()
        self.keys = keys
        self.loss_function = loss_function
        self.device = device

    @abstractmethod
    def linearize(self, x0: Variables) -> th.Tensor:
        raise

    # Forward === error
    def forward(self, x: Variables or th.Tensor) -> th.Tensor:
        return self.error(x)

    # f(x)
    @abstractmethod
    def error(self, x: Variables or th.Tensor) -> th.Tensor:
        pass

    # w(e) where e = f(x),
    def weight(self, x: Variables) -> th.Tensor:
        return self.loss_function(self.error(x))

    def _batch_jacobian(self, f, x0: th.Tensor):
        f_sum = lambda x: th.sum(f(x), axis=0) # sum over all batches
        return th.autograd.functional.jacobian(f_sum, x0, create_graph=True).swapaxes(1, 0)
