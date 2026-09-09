# CLIP + LSTM Image Captioning

A PyTorch course-project implementation of image caption generation using a pretrained OpenAI CLIP visual encoder, a learnable projection layer, and an LSTM language decoder with beam-search generation.

This repository is a cleaned portfolio implementation. Training data, model caches, checkpoints and evaluation assets are intentionally not redistributed.

## What the refactor improves

The original experiment code had several reproducibility/correctness issues that are now explicit and tested:

- CPU beam search now follows the same path as CUDA; the old code executed later decoding only inside a CUDA branch.
- Deprecated `Variable(..., volatile=True)` usage is removed.
- Caption token targets are explicitly `torch.long`.
- Training/validation image and annotation paths are required CLI inputs instead of hidden machine-specific defaults.
- Boolean CLI flags use real argparse flags instead of `type=bool`.
- The projected image recurrent output is trained to predict the **first caption token**, matching generation-time decoding; subsequent word inputs predict later words and EOS.
- Progressive CLIP fine-tuning keeps all CLIP parameters in the optimizer from the start, so later-unfrozen layers can actually update.
- New checkpoints store `state_dict + vocab + architecture config` instead of pickling a live model object.
- Lightweight tests cover tokenization, vocabulary, collation, teacher-forcing alignment, CPU beam search and reference mapping without downloading CLIP weights.

No new BLEU/CIDEr/SPICE or other quality metric is claimed by this cleanup.

## Architecture

```text
image
  |
  v
pretrained CLIP visual encoder
  |
  v
learned projection
  |
  +---- first LSTM input ----> predicts first caption token
                              |
caption token embeddings -----+
                              v
                            LSTM
                              |
                              v
                       vocabulary classifier
                              |
                              v
                         beam search
```

## Repository layout

```text
beam_search.py      CPU/CUDA-safe beam-search decoder
model.py            CLIP + projection + LSTM model and safe checkpoint format
main.py             parameterized training/validation CLI
data.py             tokenization, vocabulary, collation and COCO-style loader
predict.py          caption generation + reference matching CLI
make_coco.py        dataset-format helper
preprocessing.py    sharpening augmentation helper
utils.py            logging, optimizer construction and meters
tests/              lightweight offline tests
```

## Setup

Python 3.10+ is recommended. GPU training is strongly recommended, but the lightweight tests and beam-search unit test run on CPU.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses OpenAI CLIP from its Git repository. Install a PyTorch build appropriate for your CPU/CUDA environment if the default wheel is not suitable.

## Dataset

Training expects COCO-style annotations with `images` and `annotations` and separate train/validation image directories. Data is not included.

Example:

```text
data/
  train_images/
  val_images/
  captions_train.json
  captions_val.json
```

## Training

```bash
python main.py \
  --train-images data/train_images \
  --train-json data/captions_train.json \
  --val-images data/val_images \
  --val-json data/captions_val.json \
  --epochs 50 \
  --optimizer AdamW \
  --encoder-model ViT-B/32 \
  --keep-count 10
```

On Windows PowerShell, use one line or PowerShell backticks instead of Bash `\` continuation.

Useful flags include:

```text
--device auto|cpu|cuda
--embedding-size
--rnn-size
--num-layers
--max-length
--finetune-stage
--share-weights
--sgd-momentum / --no-sgd-momentum
--seed
```

`--share-weights` requires `embedding-size == rnn-size`.

### Teacher-forcing alignment

For a caption `[w0, w1, ..., EOS]`, training feeds:

```text
recurrent inputs: [IMAGE, w0, w1, ...]
targets:          [w0,    w1, w2, ..., EOS]
```

This matches generation, where the first word is decoded directly after the projected image feature.

## Checkpoints

The refactored checkpoint format contains:

- format version
- epoch
- vocabulary
- CLIP/LSTM architecture config
- full model `state_dict`

It does **not** pickle the live Python model object. The public inference path rejects the old pickle-style checkpoint format rather than silently loading arbitrary serialized Python objects.

## Inference

```bash
python predict.py \
  --testfolder path/to/test_images \
  --checkpoint results/checkpoint_epoch_50.pth \
  --test-json path/to/captions_test.json \
  --beam-size 3 \
  --device auto
```

The inference tool writes generated captions and matched reference captions to JSON files. Images without a matched reference are excluded from the paired output.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

The lightweight test suite does not instantiate CLIP or download model weights. It checks:

- deterministic vocabulary ordering and special tokens
- `torch.long` target IDs and EOS/padding behavior
- CPU beam-search execution
- Top-N score bookkeeping
- image-first teacher-forcing alignment
- optimizer parameter-group learning rates
- COCO reference-map behavior

## Reproducibility and limitations

- Python, NumPy and PyTorch seeds are configured by the training CLI.
- Training and validation data paths are explicit.
- CUDA mixed precision is enabled only on CUDA.
- Model quality still depends on dataset, hyperparameters and pretrained CLIP behavior.
- The repository currently does not publish a controlled caption-quality benchmark or trained checkpoint, so it should be presented as an architecture/training implementation rather than an SOTA claim.

## Scope and attribution

This is a course-project implementation and portfolio artifact, not a claim of ownership over CLIP, PyTorch, torchvision or COCO-style datasets. Third-party projects and datasets retain their own licenses and terms.

## License

MIT License. See `LICENSE`.
