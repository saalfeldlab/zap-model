"""EED-specific data loading: windows, GPU transfer, infinite batched iterators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zap_model.data.activity import ActivityData
    from zap_model.data.config import ConditionSplitConfig
    from zap_model.models.eed.config import EEDModelConfig


@dataclass
class EEDData:
    """Data iterators and metadata for EED training.

    Attributes:
        train_iter: Infinite iterator yielding batches of shape ``(B, rollout_steps+1, N)``.
        val_iter: Validation iterator (empty for now).
        num_train_windows: Total number of training windows extracted.
    """

    train_iter: Iterator[Tensor]
    val_iter: Iterator[Tensor]
    num_train_windows: int


def _infinite_shuffled_batches(
    traces: Tensor,
    starts: Tensor,
    window_size: int,
    batch_size: int,
) -> Iterator[Tensor]:
    """Yield batches by slicing windows from *traces* on the fly.

    Args:
        traces: (T, N) tensor on GPU.
        starts: 1-D int64 tensor of valid window start indices, on GPU.
        window_size: Number of frames per window (rollout_steps + 1).
        batch_size: Number of windows per batch.
    """
    num_windows = starts.shape[0]
    while True:
        perm = torch.randperm(num_windows, device=starts.device)
        for i in range(0, num_windows - batch_size + 1, batch_size):
            batch_starts = starts[perm[i : i + batch_size]]
            # (B, window_size) offsets for gather
            offsets = batch_starts.unsqueeze(1) + torch.arange(
                window_size,
                device=traces.device,
            )
            yield traces[offsets]


def make_eed_data(
    activity: ActivityData,
    splits: ConditionSplitConfig,
    model_cfg: EEDModelConfig,
    batch_size: int,
    device: torch.device,
) -> EEDData:
    """Build EED training data from loaded activity traces.

    Moves the full traces tensor to *device* once, then collects valid window
    start indices per train split. Batches are sliced on the fly — no
    materialized copy of overlapping windows.
    """
    window_size = model_cfg.rollout_steps + 1
    ranges = splits.get_ranges()

    start_list: list[int] = []
    for cond in splits.train_conditions:
        r = ranges[cond].train
        for win_start in range(r.start, r.end - window_size + 1):
            start_list.append(win_start)

    if not start_list:
        msg = "No training windows could be extracted — check splits and rollout_steps"
        raise ValueError(msg)

    traces = activity.traces.to(device)
    starts = torch.tensor(start_list, dtype=torch.long, device=device)

    train_iter = _infinite_shuffled_batches(traces, starts, window_size, batch_size)
    val_iter = iter(())

    return EEDData(
        train_iter=train_iter,
        val_iter=val_iter,
        num_train_windows=len(start_list),
    )
