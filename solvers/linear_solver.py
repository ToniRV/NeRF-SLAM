#!/usr/bin/env python3

import enum
import colored_glog as log
import torch as th

class LinearSolverType(enum.Enum):
    Inverse = 0
    Cholesky = 1
    QR = 2 # to be implemented
    MultifrontalQR = 3 # to be implemented
    CG = 4 # Conjugate Gradient, to be implemented
    # Gauss-Seidel for dense flow?
    # Use scipy.optimize...

# Linear Solver
class LinearLS:
    # loss = |Y - ([X, 1] * [w, b])|^2_sigma
    # [X, 1] == A, Y - ([X, 1] * [^w, ^b]) == b [w, b] == x
    @staticmethod
    def solve_inverse(A, b, E):
        # loss = |Ax - b|^2_sigma(\theta)
        At = th.transpose(A, 1, 2)
        AtE = At @ E
        AtEA = AtE @ A
        AtEb = AtE @ b
        x = th.linalg.inv(AtEA) @  AtEb
        return x

    # TODO(Toni): We should make the sparse version of this
    # A has size: (B, M, N) where B = batch, M = measurements, N = variables
    # w has size: (B, M, 1) and is the weight for each measurement (assumes diagonal weights)
    # b has size: (B, N)
    @staticmethod
    def solve_cholesky(A: th.Tensor, b: th.Tensor, w: th.Tensor) -> th.Tensor:
        # loss = |Ax - b|^2_w
        # Solve all Batch linear problems
        if A is None or b is None or w is None:
            return None

        # Check dimensionality, first is the batch dimension
        B, M = b.shape
        _, _, N = A.shape
        log.check_eq(th.Size([B, M, N]), A.shape)
        log.check_eq(th.Size([B, M]), w.shape) # We assume E diagonal: E=diag(w)
        WA = A * w.unsqueeze(-1) # multiply each row by the weight (uses broadcasting)
        AtWt = WA.transpose(1,2)
        AtEA = AtWt @ WA
        AtWtb = AtWt @ b.unsqueeze(-1)
        L = th.linalg.cholesky(AtEA)
        y = th.linalg.solve_triangular(L, AtWtb, upper=False)
        x = th.linalg.solve_triangular(th.transpose(L, 1, 2), y, upper=True)
        return x.squeeze(-1)

    # TODO(Toni): We should make the sparse version of this
    # A has size: (B, M, N) where B = batch, M = measurements, N = variables <- In standard form
    # b has size: (B, N)
    # @staticmethod
    # def solve_cholesky(A, b):
    #     # loss = |Ax - b|^2_2
    #     # Solve all Batch linear problems

    #     # Check dimensionality, first is the batch dimension
    #     B, M = b.shape
    #     _, _, N = A.shape
    #     log.check_eq(th.Size([B, M, N]), A.shape)
    #     At = A.transpose(1, 2) # Batched A transposes
    #     AtA = At @ A
    #     Atb = At @ b.unsqueeze(-1)
    #     L = th.linalg.cholesky(AtA)
    #     y = th.linalg.solve_triangular(L, Atb, upper=False)
    #     x = th.linalg.solve_triangular(th.transpose(L, 1, 2), y, upper=True)
    #     return x.squeeze(-1)

    # For pyth 1.11
    @staticmethod
    def solve_cholesky_11(A, b, E):
        # loss = |Ax - b|^2_sigma(\theta)
        At = th.transpose(A, 1, 2)
        AtE = At @ E
        AtEA = AtE @ A
        AtEb = AtE @ b
        L = th.linalg.cholesky(AtEA)
        x = th.linalg.cholesky_solve(AtEb, L, upper=False)
        return x

