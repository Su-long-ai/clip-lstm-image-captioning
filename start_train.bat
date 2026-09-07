@echo off
python main.py --start_epoch 0 --epochs 50 --workers 5 --optimizer AdamW --encoder_model ViT-B/32 --keep_count 10

@rem 可以将--encoder_model改成 RN50 尝试Resnet50的效果