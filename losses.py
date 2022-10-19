import torch

__all__ = ["PosAndAngleLoss", ]

from torch import nn
from torch.nn import CosineSimilarity, MSELoss


class PosAndAngleLoss(nn.Module):
    def __init__(self, alpha=100, degree=2):
        super().__init__()
        self.translational_metric = MSELoss()
        self.angular_metric = CosineSimilarity(dim=1)

    def forward(self, y_hat, y, var=0):
        return self.translational_metric(y_hat[:, 0:3], y[:, 0:3]) \
               + 1 - \
               torch.mean(self.angular_metric(y_hat[:, 3:], y[:, 3:]))
