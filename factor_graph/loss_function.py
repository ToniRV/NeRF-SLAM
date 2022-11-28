#!/usr/bin/env python3

import torch as th
from torch import nn

class LossFunction(nn.Module):
    def __init__(self, device) -> None:
        nn.Module.__init__(self)
        super(LossFunction).__init__()
        self.device = device

class CauchyLossFunction(LossFunction):
    def __init__(self, device, k_initial: float = 1.0) -> None:
        super().__init__(device)
        self.k = th.nn.Parameter(k_initial * th.ones(1, device=self.device, requires_grad=True))

    # w(e) = w(f(x))
    # The loss function weights the residuals
    def forward(self, residuals: th.Tensor) -> th.Tensor:
        return 1.0 / (1.0 + th.pow(residuals / self.k, 2))

class GMLossFunction(LossFunction):
    def __init__(self, device, k_initial: float = 1.0) -> None:
        super().__init__(device)
        self.k = th.nn.Parameter(k_initial * th.ones(1, device=self.device, requires_grad=True))

    # w(e) = w(f(x))
    # The loss function weights the residuals
    def forward(self, residuals: th.Tensor) -> th.Tensor:
        return th.pow(self.k, 2) / th.pow(self.k + th.pow(residuals, 2), 2) # Isn't this cauchy?