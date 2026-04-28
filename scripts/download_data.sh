#!/bin/bash
set -euo pipefail

DATA_DIR="/data/scene-rep/u/aquan/cv/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading ShapeNet rendered images (~7.4 GB) ==="
if [ ! -d "ShapeNetRendering" ]; then
    wget -c http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
    echo "Extracting ShapeNetRendering..."
    tar -xzf ShapeNetRendering.tgz
    rm ShapeNetRendering.tgz
else
    echo "ShapeNetRendering already exists, skipping."
fi

echo "=== Downloading ShapeNet voxelized models (~195 MB) ==="
if [ ! -d "ShapeNetVox32" ]; then
    wget -c http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
    echo "Extracting ShapeNetVox32..."
    tar -xzf ShapeNetVox32.tgz
    rm ShapeNetVox32.tgz
else
    echo "ShapeNetVox32 already exists, skipping."
fi

echo "=== Downloading train/val/test split ==="
if [ ! -f "pix2vox_splits.json" ]; then
    wget -O pix2vox_splits.json \
        "https://raw.githubusercontent.com/hzxie/Pix2Vox/master/datasets/ShapeNet.json"
else
    echo "Split file already exists, skipping."
fi

echo "=== Done. Data is in $DATA_DIR ==="
ls -lh "$DATA_DIR"
