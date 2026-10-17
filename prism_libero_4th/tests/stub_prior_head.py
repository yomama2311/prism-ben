"""TEST ONLY: small replacement for original repo prior_head.py, when unavailable."""
import torch
from torch import nn


class PriorHead(nn.Module):
    def __init__(self, z_dim, H, A_block, A_raw, hidden=64, sigma_floor=0.05):
        super().__init__()
        self.z_dim, self.H, self.A_block, self.A_raw = z_dim, H, A_block, A_raw
        self.hidden, self.sigma_floor = hidden, sigma_floor
        self.net = nn.Sequential(nn.Linear(z_dim*2, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2*H*A_block*A_raw))

    def forward(self, z, goal):
        y = self.net(torch.cat([z, goal], -1))
        mu, s = y.chunk(2, -1)
        shape = (z.shape[0], self.H, self.A_block, self.A_raw)
        return mu.view(shape), torch.nn.functional.softplus(s).view(shape) + self.sigma_floor
