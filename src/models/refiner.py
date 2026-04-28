"""3D convolutional refiner (Pix2Vox-style).

Takes a coarse 32^3 voxel grid and refines it via residual 3D convolutions.
"""

import torch
import torch.nn as nn


class VoxelRefiner(nn.Module):
    """Residual 3D-conv refiner operating on 1x32x32x32 logits."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 1, 3, padding=1),
        )

    def forward(self, coarse_logits):
        """Residual refinement: output = coarse + delta."""
        delta = self.layers(coarse_logits)
        return coarse_logits + delta
