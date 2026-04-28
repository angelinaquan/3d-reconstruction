"""3D voxel decoder: maps a feature vector to a 32x32x32 occupancy grid."""

import torch
import torch.nn as nn


class VoxelDecoder(nn.Module):
    """Transposed-conv decoder: feat_dim -> 1x32x32x32."""

    def __init__(self, feat_dim=512):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 256 * 2 * 2 * 2)

        self.deconv = nn.Sequential(
            # 256x2x2x2 -> 128x4x4x4
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # 128x4x4x4 -> 64x8x8x8
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # 64x8x8x8 -> 32x16x16x16
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 32x16x16x16 -> 1x32x32x32
            nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 256, 2, 2, 2)
        x = self.deconv(x)
        return x  # raw logits, (B, 1, 32, 32, 32)
