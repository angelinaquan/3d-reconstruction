#!/bin/bash
#SBATCH --job-name=cv3d_eval
#SBATCH --partition=vision-shared-l40s,vision-shared-a100,vision-shared-h100,vision-shared-h200,csail-shared-h200
#SBATCH --account=vision-sitzmann
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /data/scene-rep/u/aquan/cv/logs/%x_%j.out
#SBATCH -e /data/scene-rep/u/aquan/cv/logs/%x_%j.err
set -euo pipefail

PYTHON=/data/scene-rep/u/gcaz/miniconda3/envs/drifting_dd/bin/python
PROJECT=/data/scene-rep/u/aquan/cv
export PYTHONUNBUFFERED=1

cd "$PROJECT"

exec $PYTHON src/evaluate.py "$@"
