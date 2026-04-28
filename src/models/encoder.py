"""Image encoder: pretrained ResNet backbone with final FC stripped."""

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_encoder(name='resnet18', pretrained=True):
    """Return (encoder_module, feature_dim)."""
    weights_map = {
        'resnet18': (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT, 512),
        'resnet50': (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT, 2048),
    }
    factory, weights, feat_dim = weights_map[name]
    backbone = factory(weights=weights if pretrained else None)
    # Drop avgpool + fc; keep everything through layer4
    encoder = nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return encoder, feat_dim
