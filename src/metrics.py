"""Evaluation metrics for voxel reconstruction."""

import torch
import numpy as np
from collections import defaultdict


def voxel_iou(pred_logits, targets, threshold=0.5):
    """Compute per-sample IoU between predicted logits and binary GT voxels.

    Returns a tensor of shape (B,) with IoU for each sample.
    """
    pred = (torch.sigmoid(pred_logits) >= threshold).float()
    targets = (targets >= threshold).float()

    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = targets.view(targets.size(0), -1)

    intersection = (pred_flat * tgt_flat).sum(dim=1)
    union = ((pred_flat + tgt_flat) >= 1).float().sum(dim=1)

    iou = intersection / union.clamp(min=1e-6)
    return iou


class IoUTracker:
    """Accumulate per-category and overall IoU across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cat_ious = defaultdict(list)
        self.all_ious = []

    def update(self, pred_logits, targets, taxonomy_ids):
        ious = voxel_iou(pred_logits, targets).cpu().numpy()
        for iou_val, tid in zip(ious, taxonomy_ids):
            self.cat_ious[tid].append(float(iou_val))
            self.all_ious.append(float(iou_val))

    def compute(self):
        results = {}
        for tid, vals in sorted(self.cat_ious.items()):
            results[tid] = np.mean(vals)
        results['overall'] = np.mean(self.all_ious) if self.all_ious else 0.0
        return results
