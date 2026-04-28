"""Training script for single-image 3D voxel reconstruction."""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import get_dataloader, TAXONOMY_NAMES
from src.models import ReconNet
from src.losses import build_loss
from src.metrics import voxel_iou, IoUTracker


def parse_args():
    p = argparse.ArgumentParser(description='Train 3D reconstruction model')
    p.add_argument('--config', type=str, default='configs/default.yaml')
    # Override any config value from CLI
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--encoder', type=str, default=None, choices=['resnet18', 'resnet50'])
    p.add_argument('--use_refiner', type=int, default=None, choices=[0, 1])
    p.add_argument('--loss', type=str, default=None, choices=['bce', 'dice', 'focal', 'bce_dice'])
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return p.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root is not None:
        cfg['data']['root'] = args.data_root
    if args.encoder is not None:
        cfg['model']['encoder'] = args.encoder
    if args.use_refiner is not None:
        cfg['model']['use_refiner'] = bool(args.use_refiner)
    if args.loss is not None:
        cfg['train']['loss'] = args.loss
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['train']['lr'] = args.lr
    if args.seed is not None:
        cfg['train']['seed'] = args.seed
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, use_refiner):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for imgs, voxels, tids, mids in tqdm(loader, desc='train', leave=False):
        imgs = imgs.to(device)
        voxels = voxels.to(device)

        refined, coarse = model(imgs)

        loss = criterion(refined, voxels)
        if use_refiner:
            loss = loss + criterion(coarse, voxels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_iou += voxel_iou(refined.detach(), voxels).sum().item()
        n += bs

    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tracker = IoUTracker()
    total_loss = 0.0
    n = 0

    for imgs, voxels, tids, mids in tqdm(loader, desc='eval', leave=False):
        imgs = imgs.to(device)
        voxels = voxels.to(device)

        refined, _ = model(imgs)
        loss = criterion(refined, voxels)

        total_loss += loss.item() * imgs.size(0)
        tracker.update(refined, voxels, tids)
        n += imgs.size(0)

    cat_ious = tracker.compute()
    return total_loss / n, cat_ious


def main():
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg['train']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # Data
    train_loader = get_dataloader(
        cfg['data']['root'], 'train',
        img_size=cfg['data']['img_size'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['data']['num_workers'],
    )
    val_loader = get_dataloader(
        cfg['data']['root'], 'val',
        img_size=cfg['data']['img_size'],
        batch_size=cfg['eval']['batch_size'],
        num_workers=cfg['data']['num_workers'],
    )

    # Model
    model = ReconNet(
        encoder_name=cfg['model']['encoder'],
        pretrained=cfg['model']['pretrained'],
        use_refiner=cfg['model']['use_refiner'],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {param_count:,}', flush=True)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['train']['epochs'],
    )

    criterion = build_loss(cfg['train']['loss'])

    # Resume
    start_epoch = 0
    best_iou = 0.0
    os.makedirs(cfg['checkpoint']['dir'], exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_iou = ckpt.get('best_iou', 0.0)
        print(f'Resumed from epoch {start_epoch}, best IoU {best_iou:.4f}', flush=True)

    # Run name
    refiner_tag = 'refiner' if cfg['model']['use_refiner'] else 'noref'
    run_name = args.run_name or f"{cfg['model']['encoder']}_{refiner_tag}_{cfg['train']['loss']}_lr{cfg['train']['lr']}_bs{cfg['train']['batch_size']}"

    wandb.init(
        project=cfg['wandb']['project'],
        entity=cfg['wandb'].get('entity'),
        name=run_name,
        config=cfg,
        resume='allow',
        settings=wandb.Settings(init_timeout=300),
    )

    # Training loop
    for epoch in range(start_epoch, cfg['train']['epochs']):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_refiner=cfg['model']['use_refiner'],
        )
        scheduler.step()

        log = {
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': train_loss,
            'train_iou': train_iou,
        }

        # Validate periodically
        do_eval = (
            (epoch + 1) % cfg['checkpoint']['save_every'] == 0
            or epoch == 0
            or epoch == cfg['train']['epochs'] - 1
        )
        if do_eval:
            val_loss, cat_ious = evaluate(model, val_loader, criterion, device)
            overall_iou = cat_ious.pop('overall')
            log['val_loss'] = val_loss
            log['val_iou'] = overall_iou

            for tid, iou_val in cat_ious.items():
                name = TAXONOMY_NAMES.get(tid, tid)
                log[f'val_iou/{name}'] = iou_val

            print(
                f'Epoch {epoch+1}/{cfg["train"]["epochs"]}  '
                f'train_loss={train_loss:.4f}  train_iou={train_iou:.4f}  '
                f'val_loss={val_loss:.4f}  val_iou={overall_iou:.4f}  '
                f'lr={scheduler.get_last_lr()[0]:.6f}',
                flush=True,
            )

            ckpt_data = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_iou': max(best_iou, overall_iou),
                'config': cfg,
            }
            ckpt_path = os.path.join(cfg['checkpoint']['dir'], f'{run_name}_latest.pt')
            torch.save(ckpt_data, ckpt_path)

            if overall_iou > best_iou:
                best_iou = overall_iou
                best_path = os.path.join(cfg['checkpoint']['dir'], f'{run_name}_best.pt')
                torch.save(ckpt_data, best_path)
                print(f'  New best IoU: {best_iou:.4f}', flush=True)
        else:
            print(
                f'Epoch {epoch+1}/{cfg["train"]["epochs"]}  '
                f'train_loss={train_loss:.4f}  train_iou={train_iou:.4f}  '
                f'lr={scheduler.get_last_lr()[0]:.6f}',
                flush=True,
            )

        wandb.log(log)

    print(f'\nTraining complete. Best val IoU: {best_iou:.4f}', flush=True)
    wandb.finish()


if __name__ == '__main__':
    main()
