from __future__ import annotations

import torch

from main import teacher_forcing_batch
from utils import AverageMeter, select_optimizer


def test_teacher_forcing_aligns_image_first_prediction_with_full_target() -> None:
    captions = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    input_captions, input_lengths, targets, target_lengths = teacher_forcing_batch(captions, [3, 2])
    assert input_captions.tolist() == [[1, 2], [4, 5]]
    assert input_lengths == [2, 1]
    assert torch.equal(targets, captions)
    assert target_lengths == [3, 2]


def test_average_meter_tracks_weighted_average() -> None:
    meter = AverageMeter()
    meter.update(2.0, n=2)
    meter.update(5.0, n=1)
    assert meter.avg == 3.0


def test_select_optimizer_creates_three_parameter_groups() -> None:
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    p3 = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = select_optimizer(
        "AdamW",
        clip_param_group=([p1], 1e-5),
        lstm_param_group=([p2], 1e-3),
        other_param_group=([p3], 2e-3),
    )
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-3, 2e-3]
