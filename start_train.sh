#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --train-images /path/to/train/images \
  --train-json /path/to/captions_train.json \
  --val-images /path/to/val/images \
  --val-json /path/to/captions_val.json \
  --epochs 50 \
  --workers 4 \
  --optimizer AdamW \
  --encoder-model ViT-B/32 \
  --keep-count 10
