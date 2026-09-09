@echo off
python main.py ^
  --train-images C:\path\to\train\images ^
  --train-json C:\path\to\captions_train.json ^
  --val-images C:\path\to\val\images ^
  --val-json C:\path\to\captions_val.json ^
  --epochs 50 ^
  --workers 4 ^
  --optimizer AdamW ^
  --encoder-model ViT-B/32 ^
  --keep-count 10
