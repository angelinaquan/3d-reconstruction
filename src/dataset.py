"""ShapeNet / R2N2 dataset loader for single-image 3D voxel reconstruction.

Expected directory layout after running scripts/download_data.sh:

    data/
      ShapeNetRendering/<taxonomy_id>/<model_id>/rendering/{00..23}.png
      ShapeNetVox32/<taxonomy_id>/<model_id>/model.binvox
      pix2vox_splits.json
"""

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import binvox_rw

TAXONOMY_NAMES = {
    '02691156': 'aeroplane',
    '02828884': 'bench',
    '02933112': 'cabinet',
    '02958343': 'car',
    '03001627': 'chair',
    '03211117': 'display',
    '03636649': 'lamp',
    '03691459': 'speaker',
    '04090263': 'rifle',
    '04256520': 'sofa',
    '04379243': 'table',
    '04401088': 'telephone',
    '04530566': 'watercraft',
}


class ShapeNetDataset(Dataset):
    """Single-view ShapeNet dataset that returns one random rendering + voxels."""

    def __init__(self, data_root, split='train', img_size=224, random_view=True):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.random_view = random_view and (split == 'train')

        self.rendering_root = os.path.join(data_root, 'ShapeNetRendering')
        self.voxel_root = os.path.join(data_root, 'ShapeNetVox32')

        splits_path = os.path.join(data_root, 'pix2vox_splits.json')
        with open(splits_path) as f:
            taxonomy_list = json.load(f)

        self.samples = []
        for tax in taxonomy_list:
            tid = tax['taxonomy_id']
            for model_id in tax.get(split, []):
                voxel_path = os.path.join(self.voxel_root, tid, model_id, 'model.binvox')
                render_dir = os.path.join(self.rendering_root, tid, model_id, 'rendering')
                if os.path.isfile(voxel_path) and os.path.isdir(render_dir):
                    self.samples.append((tid, model_id, render_dir, voxel_path))

        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print(f'[{split}] Loaded {len(self.samples)} samples from {len(TAXONOMY_NAMES)} categories')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tid, model_id, render_dir, voxel_path = self.samples[idx]

        pngs = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
        if self.random_view:
            view_file = random.choice(pngs)
        else:
            view_file = pngs[0]

        img = Image.open(os.path.join(render_dir, view_file)).convert('RGB')
        img_tensor = self.img_transform(img)

        with open(voxel_path, 'rb') as f:
            voxel = binvox_rw.read_as_3d_array(f)
        voxel_tensor = torch.from_numpy(voxel.data.copy()).float().unsqueeze(0)  # (1, 32, 32, 32)

        return img_tensor, voxel_tensor, tid, model_id


def get_dataloader(data_root, split, img_size=224, batch_size=32, num_workers=8):
    random_view = (split == 'train')
    ds = ShapeNetDataset(data_root, split=split, img_size=img_size, random_view=random_view)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
