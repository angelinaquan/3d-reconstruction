"""Full reconstruction network: encoder + decoder + optional refiner."""

import torch
import torch.nn as nn

from .encoder import build_encoder
from .decoder import VoxelDecoder
from .refiner import VoxelRefiner


class ReconNet(nn.Module):
    def __init__(self, encoder_name='resnet18', pretrained=True, use_refiner=True):
        super().__init__()
        self.encoder, feat_dim = build_encoder(encoder_name, pretrained)
        self.decoder = VoxelDecoder(feat_dim)
        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = VoxelRefiner()

    def forward(self, img):
        feat = self.encoder(img)
        coarse = self.decoder(feat)
        if self.use_refiner:
            refined = self.refiner(coarse)
            return refined, coarse
        return coarse, coarse
