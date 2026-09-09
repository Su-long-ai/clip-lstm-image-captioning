from __future__ import annotations

import logging
from typing import Iterable

import torch
from rich.logging import RichHandler


def setup_logging(log_file: str = "log.txt", level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
        force=True,
    )
    console = RichHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)


def select_optimizer(
    optimizer_name: str,
    clip_param_group: tuple[Iterable[torch.nn.Parameter], float],
    lstm_param_group: tuple[Iterable[torch.nn.Parameter], float],
    other_param_group: tuple[Iterable[torch.nn.Parameter], float],
    *,
    momentum: float = 0.9,
    alpha: float = 0.99,
    enable_sgd_momentum: bool = True,
    weight_decay: float = 1e-4,
):
    groups = [
        {"params": list(clip_param_group[0]), "lr": clip_param_group[1]},
        {"params": list(lstm_param_group[0]), "lr": lstm_param_group[1]},
        {"params": list(other_param_group[0]), "lr": other_param_group[1]},
    ]
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=momentum if enable_sgd_momentum else 0.0,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, momentum=momentum, alpha=alpha, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose SGD, Adam, AdamW or RMSprop.")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count
