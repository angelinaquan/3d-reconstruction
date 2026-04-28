"""Visualize voxel predictions as 3D plots.

Can be run standalone to visualize a checkpoint's predictions, or imported
to generate figures for the paper.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def plot_voxel(voxel_grid, ax=None, title=None, color='steelblue', alpha=0.6):
    """Render a 3D boolean/float voxel grid using matplotlib voxels."""
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.cpu().numpy()

    while voxel_grid.ndim > 3:
        voxel_grid = voxel_grid.squeeze(0)

    filled = voxel_grid > 0.5
    colors_arr = np.empty(filled.shape, dtype=object)
    colors_arr[filled] = color

    ax.voxels(filled, facecolors=colors_arr, edgecolor='none', alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    return ax


def make_comparison_figure(input_img, gt_voxel, pred_voxel, save_path=None):
    """Side-by-side: input image | GT voxels | predicted voxels."""
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    if isinstance(input_img, torch.Tensor):
        img_np = input_img.cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np.transpose(1, 2, 0) * std + mean
            img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(input_img)
    ax1.imshow(img_np)
    ax1.set_title('Input')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_voxel(gt_voxel, ax=ax2, title='Ground Truth', color='forestgreen')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_voxel(pred_voxel, ax=ax3, title='Prediction', color='steelblue')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.close(fig)
    return fig


@torch.no_grad()
def visualize_checkpoint(checkpoint_path, data_root, split='test', n_samples=8, out_dir=None):
    """Load a checkpoint and produce comparison figures for random samples."""
    from src.dataset import ShapeNetDataset, TAXONOMY_NAMES
    from src.models import ReconNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    model = ReconNet(
        encoder_name=cfg['model']['encoder'],
        pretrained=False,
        use_refiner=cfg['model']['use_refiner'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = ShapeNetDataset(data_root, split=split, img_size=cfg['data']['img_size'], random_view=False)

    indices = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)
    out_dir = out_dir or 'visualizations'
    os.makedirs(out_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        img, voxel_gt, tid, mid = ds[idx]
        img_gpu = img.unsqueeze(0).to(device)
        refined, _ = model(img_gpu)
        pred = (torch.sigmoid(refined) >= 0.5).float()

        cat_name = TAXONOMY_NAMES.get(tid, tid)
        save_path = os.path.join(out_dir, f'{cat_name}_{mid[:8]}_{i}.png')
        make_comparison_figure(img, voxel_gt, pred.squeeze(0), save_path=save_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_root', type=str, default='/data/scene-rep/u/aquan/cv/data')
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--n_samples', type=int, default=8)
    p.add_argument('--out_dir', type=str, default='visualizations')
    args = p.parse_args()

    visualize_checkpoint(
        args.checkpoint, args.data_root,
        split=args.split, n_samples=args.n_samples, out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
