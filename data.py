from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Iterable

import torch

PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"
EOS_TOKEN = "EOS"
DEFAULT_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def simple_tokenize(captions: Iterable[str]) -> list[list[str]]:
    processed: list[list[str]] = []
    for caption in captions:
        text = str(caption).lower()
        for punct in string.punctuation:
            text = text.replace(punct, f" {punct} ")
        processed.append(text.strip().split())
    return processed


def build_vocab_from_captions(captions: Iterable[str], *, num_words: int = 10_000) -> list[str]:
    if num_words <= 0:
        raise ValueError("num_words must be positive")
    counts: dict[str, int] = {}
    for tokens in simple_tokenize(captions):
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocabulary = [token for token, _ in ranked[:num_words]]
    return [PAD_TOKEN, *vocabulary, UNK_TOKEN, EOS_TOKEN]


def build_vocab(ann_file: str | Path, *, num_words: int = 10_000) -> list[str]:
    path = Path(ann_file)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    captions = [str(item["caption"]) for item in payload.get("annotations", []) if "caption" in item]
    if not captions:
        raise ValueError(f"No caption annotations found in {path}")
    return build_vocab_from_captions(captions, num_words=num_words)


class CaptionProcessor:
    def __init__(self, vocab: list[str], rnd_caption: bool = True):
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.unk = self.word2idx[UNK_TOKEN]
        self.rnd_caption = rnd_caption

    def __call__(self, captions: list[str]) -> torch.Tensor:
        tokenized = simple_tokenize(captions)
        if not tokenized:
            raise ValueError("At least one caption is required")
        tokens = random.choice(tokenized) if self.rnd_caption else tokenized[0]
        return torch.tensor([self.word2idx.get(word, self.unk) for word in tokens], dtype=torch.long)


class BatchCollator:
    def __init__(self, vocab: list[str], max_length: int = 50):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.padding = vocab.index(PAD_TOKEN)
        self.eos = vocab.index(EOS_TOKEN)
        self.max_length = max_length

    def __call__(self, img_cap):
        if not img_cap:
            raise ValueError("Cannot collate an empty batch")
        ordered = sorted(img_cap, key=lambda pair: len(pair[1]), reverse=True)
        imgs, caps = zip(*ordered, strict=True)
        imgs = torch.stack(list(imgs), dim=0)
        lengths = [min(len(caption) + 1, self.max_length) for caption in caps]
        batch_length = max(lengths)
        cap_tensor = torch.full((len(caps), batch_length), self.padding, dtype=torch.long)

        for index, caption in enumerate(caps):
            token_count = lengths[index] - 1
            if token_count:
                cap_tensor[index, :token_count].copy_(caption[:token_count].long())
            cap_tensor[index, token_count] = self.eos
        return imgs, (cap_tensor, lengths)


def create_target(vocab: list[str], rnd_caption: bool = True):
    processor = CaptionProcessor(vocab, rnd_caption=rnd_caption)
    return processor


def create_batches(vocab: list[str], max_length: int = 50):
    return BatchCollator(vocab, max_length=max_length)


def get_coco_data(
    vocab: list[str],
    *,
    root: str | Path,
    ann_file: str | Path,
    train: bool,
    img_size: int = 224,
    scale_size: int = 256,
    normalize: dict = DEFAULT_NORMALIZE,
    target_img_size=None,
):
    """Build a torchvision CocoCaptions dataset.

    torchvision and pycocotools remain runtime-only dependencies and are
    imported here so tokenization/vocabulary tests stay lightweight.
    """

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    from preprocessing import Sharpen

    transform_steps = []
    if target_img_size:
        transform_steps.append(transforms.Resize(target_img_size))
    elif train:
        transform_steps.extend([transforms.Resize(scale_size), transforms.RandomCrop(img_size)])
    else:
        transform_steps.extend([transforms.Resize(scale_size), transforms.CenterCrop(img_size)])

    if train:
        transform_steps.extend(
            [
                transforms.RandomApply([Sharpen(factor=1.5)], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3),
            ]
        )
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(**normalize)])

    dataset = dset.CocoCaptions(
        root=str(root),
        annFile=str(ann_file),
        transform=transforms.Compose(transform_steps),
        target_transform=CaptionProcessor(vocab, rnd_caption=train),
    )
    return dataset, vocab


def get_iterator(
    data,
    *,
    batch_size: int = 32,
    max_length: int = 30,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    dataset, vocab = data
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=BatchCollator(vocab, max_length=max_length),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

