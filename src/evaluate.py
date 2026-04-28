"""Evaluate a trained model: compute per-category IoU on val or test split."""

import argparse
import os
import sys
import json

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import get_dataloader, TAXONOMY_NAMES
from src.models import ReconNet
from src.losses import build_loss
from src.metrics import IoUTracker


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate 3D reconstruction model')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    p.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']

    model = ReconNet(
        encoder_name=cfg['model']['encoder'],
        pretrained=False,
        use_refiner=cfg['model']['use_refiner'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    loader = get_dataloader(
        cfg['data']['root'], args.split,
        img_size=cfg['data']['img_size'],
        batch_size=cfg['eval']['batch_size'],
        num_workers=cfg['data']['num_workers'],
    )

    criterion = build_loss(cfg['train']['loss'])
    tracker = IoUTracker()
    total_loss = 0.0
    n = 0

    for imgs, voxels, tids, mids in tqdm(loader, desc=f'eval ({args.split})'):
        imgs = imgs.to(device)
        voxels = voxels.to(device)

        refined, _ = model(imgs)
        loss = criterion(refined, voxels)
        total_loss += loss.item() * imgs.size(0)
        tracker.update(refined, voxels, tids)
        n += imgs.size(0)

    cat_ious = tracker.compute()
    overall = cat_ious.pop('overall')

    print(f'\n{"Category":<20s} {"IoU":>8s}  {"Count":>6s}')
    print('-' * 38)
    for tid in sorted(cat_ious.keys()):
        name = TAXONOMY_NAMES.get(tid, tid)
        count = len(tracker.cat_ious[tid])
        print(f'{name:<20s} {cat_ious[tid]:8.4f}  {count:6d}')
    print('-' * 38)
    print(f'{"Overall":<20s} {overall:8.4f}  {n:6d}')
    print(f'Loss: {total_loss / n:.4f}')

    if args.output:
        results = {
            'split': args.split,
            'overall_iou': overall,
            'loss': total_loss / n,
            'per_category': {TAXONOMY_NAMES.get(t, t): v for t, v in cat_ious.items()},
            'checkpoint': args.checkpoint,
        }
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
