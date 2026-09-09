from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence

from data import build_vocab, get_coco_data, get_iterator
from utils import AverageMeter, select_optimizer, setup_logging

console = Console()

ENCODERS = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CLIP + projection + LSTM image captioning")
    parser.add_argument("--train-images", type=Path, required=True, help="Training image directory")
    parser.add_argument("--train-json", type=Path, required=True, help="COCO-style training annotations")
    parser.add_argument("--val-images", type=Path, required=True, help="Validation image directory")
    parser.add_argument("--val-json", type=Path, required=True, help="COCO-style validation annotations")
    parser.add_argument("--encoder-model", "--encoder_model", default="ViT-B/32", choices=ENCODERS)
    parser.add_argument("--embedding-size", "--embedding_size", default=256, type=int)
    parser.add_argument("--rnn-size", "--rnn_size", default=256, type=int)
    parser.add_argument("--num-layers", "--num_layers", default=2, type=int)
    parser.add_argument("--max-length", "--max_length", default=30, type=int)
    parser.add_argument("--finetune-stage", "--finetune_stage", default=4, type=int)
    parser.add_argument("--start-epoch", "--start_epoch", default=0, type=int)
    parser.add_argument("--optimizer", default="AdamW", choices=["SGD", "Adam", "AdamW", "RMSprop"])
    parser.add_argument("--grad-clip", "--grad_clip", default=6.0, type=float)
    parser.add_argument("--share-weights", "--share_weights", action="store_true")
    parser.add_argument("-j", "--workers", default=0, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", default=32, type=int)
    parser.add_argument("--lr", "--learning-rate", default=5e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float)
    parser.add_argument("--alpha", default=0.99, type=float)
    parser.add_argument("--sgd-momentum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-freq", "-p", default=20, type=int)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--save", default="")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--keep-count", "-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def teacher_forcing_batch(captions: torch.Tensor, lengths: list[int]):
    """Align image-first recurrent inputs with full caption targets.

    Input text excludes EOS. The recurrent output produced after the image
    predicts the first token, so the packed target remains the full caption,
    including EOS.
    """

    if captions.dtype != torch.long:
        raise TypeError("captions must be torch.long")
    if any(length <= 0 for length in lengths):
        raise ValueError("caption lengths must be positive")
    return captions[:, :-1], [length - 1 for length in lengths], captions, lengths


def run_epoch(model, data, *, device: torch.device, training: bool, optimizer, grad_clip: float, print_freq: int, epoch: int) -> float:
    loss_fn = nn.CrossEntropyLoss()
    perplexity = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    timestamp = time.time()

    for batch_idx, (images, (captions, lengths)) in enumerate(data):
        data_time.update(time.time() - timestamp)
        images = images.to(device, non_blocking=use_amp)
        captions = captions.to(device, non_blocking=use_amp)
        input_captions, input_lengths, target_captions, target_lengths = teacher_forcing_batch(captions, lengths)

        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with amp_context:
            predictions, _ = model(images, input_captions, input_lengths)
            packed_targets = pack_padded_sequence(
                target_captions,
                target_lengths,
                batch_first=True,
                enforce_sorted=True,
            )
            error = loss_fn(predictions, packed_targets.data)

        if training:
            scaler.scale(error).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        perplexity.update(math.exp(min(error.item(), 20.0)), n=images.shape[0])
        batch_time.update(time.time() - timestamp)
        timestamp = time.time()
        if batch_idx % print_freq == 0:
            logging.info(
                "%s epoch=%s batch=%s/%s time=%.3f data=%.3f perplexity=%.4f",
                "train" if training else "val",
                epoch,
                batch_idx,
                len(data),
                batch_time.val,
                data_time.val,
                perplexity.val,
            )
    return perplexity.avg


def train_model(
    *,
    start_epoch: int,
    epochs: int,
    model,
    optimizer,
    train_data,
    val_data,
    checkpoint_pattern: str,
    save_path: str,
    keep_count: int,
    finetune_stage: int,
    base_clip_lr: float,
    device: torch.device,
    print_freq: int,
    grad_clip: float,
) -> None:
    for epoch in range(start_epoch, epochs):
        model.train()
        model.finetune_clip(
            optimizer=optimizer,
            current_epoch=epoch,
            total_epochs=epochs,
            base_clip_lr=base_clip_lr,
            stages=finetune_stage,
        )
        train_perplexity = run_epoch(
            model,
            train_data,
            device=device,
            training=True,
            optimizer=optimizer,
            grad_clip=grad_clip,
            print_freq=print_freq,
            epoch=epoch + 1,
        )
        model.eval()
        with torch.no_grad():
            val_perplexity = run_epoch(
                model,
                val_data,
                device=device,
                training=False,
                optimizer=optimizer,
                grad_clip=grad_clip,
                print_freq=print_freq,
                epoch=epoch + 1,
            )
        logging.info(
            "epoch=%s train_perplexity=%.4f val_perplexity=%.4f",
            epoch + 1,
            train_perplexity,
            val_perplexity,
        )
        model.save_checkpoint(checkpoint_pattern % (epoch + 1), save_path, keep_count, epoch + 1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.start_epoch < 0 or args.start_epoch >= args.epochs:
        raise ValueError("start-epoch must be in [0, epochs)")

    set_seed(args.seed)
    device = resolve_device(args.device)
    save_path = args.results_dir / args.save
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(str(save_path / "log.txt"), level=logging.DEBUG if args.debug else logging.INFO)

    vocab = build_vocab(args.train_json, num_words=args.vocab_size)
    from model import CaptionClipModel

    model = CaptionClipModel(
        vocab,
        args.encoder_model,
        embedding_size=args.embedding_size,
        rnn_size=args.rnn_size,
        num_layers=args.num_layers,
        share_embedding_weights=args.share_weights,
        use_cuda=device.type == "cuda",
    ).to(device)

    train_data = get_iterator(
        get_coco_data(vocab, root=args.train_images, ann_file=args.train_json, train=True),
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_data = get_iterator(
        get_coco_data(vocab, root=args.val_images, ann_file=args.val_json, train=False),
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    clip_params = list(model.clip_model.parameters())
    lstm_params = list(model.rnn.parameters())
    reserved = {id(parameter) for parameter in [*clip_params, *lstm_params]}
    other_params = [parameter for parameter in model.parameters() if id(parameter) not in reserved]
    clip_lr = args.lr * 0.005
    optimizer = select_optimizer(
        args.optimizer,
        clip_param_group=(clip_params, clip_lr),
        lstm_param_group=(lstm_params, args.lr),
        other_param_group=(other_params, args.lr),
        momentum=args.momentum,
        alpha=args.alpha,
        enable_sgd_momentum=args.sgd_momentum,
        weight_decay=args.weight_decay,
    )

    console.log(f"device={device} vocab={len(vocab)} train={len(train_data.dataset)} val={len(val_data.dataset)}")
    train_model(
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        checkpoint_pattern=str(save_path / "checkpoint_epoch_%s.pth"),
        save_path=str(save_path),
        keep_count=args.keep_count,
        finetune_stage=args.finetune_stage,
        base_clip_lr=clip_lr,
        device=device,
        print_freq=args.print_freq,
        grad_clip=args.grad_clip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
