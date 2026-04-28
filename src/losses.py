"""Loss functions for voxel reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary voxel grids."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in voxel grids
    (most voxels are empty)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def build_loss(name='bce'):
    """Return a loss function by name."""
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif name == 'dice':
        return DiceLoss()
    elif name == 'focal':
        return FocalLoss()
    elif name == 'bce_dice':
        bce = nn.BCEWithLogitsLoss()
        dice = DiceLoss()
        return lambda logits, targets: bce(logits, targets) + dice(logits, targets)
    else:
        raise ValueError(f'Unknown loss: {name}')
