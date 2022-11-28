#!/usr/bin/env python3

import torch as th
from torch.utils.data import DataLoader

from slam.meta_slam import MetaSLAM

class MetaSolver:
    def __init__(self, epochs: int, optimizer: th.optim.Optimizer, meta_slam: MetaSLAM):
        self.epochs = epochs
        self.optimizer = optimizer
        self.meta_slam = meta_slam
        self.log_every_n = 1

    def solve(self, dataloader: DataLoader):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}\n-------------------------------")
            for i, mini_batch in enumerate(dataloader):
                print(f"Batch {i}\n-------------------------------")

                # Build and solve factor graphs
                x = self.meta_slam(mini_batch)
                if self.logging_callback:
                    self.logging_callback()

                # Compute and print loss
                if self.optimizer:
                    loss = self.meta_slam.calculate_loss(x, mini_batch)
                    if i % self.log_every_n == 0:
                        print(f"outer_loss: {loss.item():>7f}  [{i:>5d}/{len(dataloader):>5d}]")

                    # Zero gradients, perform a backward pass, and update the weights from the outer optimization
                    self.optimizer.zero_grad() 
                    loss.backward()
                    self.optimizer.step()

    def register_logging_callback(self, log):
        self.logging_callback = log