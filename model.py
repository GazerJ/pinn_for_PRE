"""Checkpoint-compatible hard-boundary network, extracted from PhaseK2to8.py."""
import numpy as np
import torch
import torch as th
from torch import nn

class HardConstraintPINN(nn.Module):

    def __init__(self):
        super(HardConstraintPINN, self).__init__()
        self.net = nn.Sequential(nn.Linear(5, 800), nn.Tanh(), nn.Linear(800, 800), nn.Tanh(), nn.Linear(800, 800), nn.Tanh(), nn.Linear(800, 800), nn.Tanh(), nn.Linear(800, 8))
        for i in range(1, 11):
            setattr(self, f'sigma{i}', nn.Parameter(torch.tensor(1.0)))

    def forward(self, x):
        y = self.net(x)
        (theta, phi, Ttheta, Tphi, lx, ly, lz, l1) = (y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6], y[:, 6:7], y[:, 7:8])
        x_coord = x[:, 0]
        Titheta = x[:, -1]
        D0 = x_coord
        D1 = 1 - x_coord
        y = [np.pi / 2 + D0 * theta[:, 0], np.pi + D0 * phi[:, 0], Titheta + D0 * Ttheta[:, 0], np.pi / 2 + D0 * Tphi[:, 0], D1 * lx[:, 0], D1 * ly[:, 0], D1 * lz[:, 0], D1 * l1[:, 0]]
        return th.stack(y).T
