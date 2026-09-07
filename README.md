# CLIP + LSTM Image Captioning

A PyTorch course-project implementation of image caption generation using a pretrained OpenAI CLIP visual encoder, a learnable projection layer, and an LSTM language decoder with beam-search generation.

This repository is a cleaned, source-focused version of the original experiment directory. Training data, pretrained weights, checkpoints, evaluation assets, and other large/generated files are intentionally not included.

## Highlights

- CLIP visual encoder (`ViT-B/32`, RN50 and other CLIP backbones supported by the training CLI)
- Learnable projection from CLIP image features into the decoder embedding space
- Multi-layer LSTM caption decoder
- Beam-search caption generation
- Progressive CLIP fine-tuning across training stages
- Mixed-precision CUDA training
- Checkpoint rotation to limit disk usage
- COCO-style caption data loading and vocabulary construction

## Repository layout

```text
beam_search.py      Beam-search decoder
model.py            CLIP + projection + LSTM caption model
main.py             Training and validation loop
data.py             Vocabulary and COCO-style data loading
predict.py          Caption generation CLI
make_coco.py        Dataset-format conversion helper
preprocessing.py    Preprocessing helpers
utils.py            Logging, meters and optimizer helpers
start_train.sh      Example Linux training command
start_train.bat     Example Windows training command
requirements.txt    Python dependencies
```

## Setup

Python 3.10+ is recommended. GPU training is strongly recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The project uses the official OpenAI CLIP package through the Git dependency in `requirements.txt`.

## Dataset

The original experiments used a COCO-style caption dataset. Dataset files are not redistributed here.

Expected inputs include:

- an image directory
- a COCO-style annotations JSON containing `images` and `annotations`
- train/validation data prepared for the loaders in `data.py`

Keep datasets outside Git or in a path ignored by `.gitignore`.

## Training

Example:

```bash
python main.py \
  --start_epoch 0 \
  --epochs 50 \
  --workers 5 \
  --optimizer AdamW \
  --encoder_model ViT-B/32 \
  --keep_count 10
```

The training code supports progressive CLIP unfreezing through `--finetune_stage` and configurable embedding/RNN dimensions, learning rate, optimizer and checkpoint retention.

## Inference

`predict.py` was cleaned up to use explicit CLI paths instead of machine-specific hardcoded paths.

```bash
python predict.py \
  --testfolder path/to/test_images \
  --checkpoint results/checkpoint_epoch_50.pth \
  --test-json path/to/captions_test.json \
  --beam-size 3
```

Generated captions are written to `gen_testcaption.json` by default, with matched reference captions written to `refer_testcaption.json`.

## Scope and attribution

This is a course-project implementation and portfolio artifact, not a claim of ownership over CLIP, PyTorch, torchvision, COCO-style datasets, or third-party evaluation toolkits. Those projects and datasets retain their own licenses and terms. Large third-party assets and bundled evaluation code are deliberately excluded from this repository.

## Notes

The original experiment folder contained datasets, pretrained CLIP weights, checkpoints, generated outputs, and unrelated research assets. They are intentionally excluded so this repository stays focused on the implementation itself.

## License

MIT License. See `LICENSE`. Third-party libraries, pretrained models and datasets retain their own licenses and terms.
