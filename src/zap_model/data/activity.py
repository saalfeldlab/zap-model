"""Shared activity data loading — disk → CPU tensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from zap_model.data.zarr import read_array

if TYPE_CHECKING:
    import numpy as np

    from zap_model.data.config import ActivityConfig


def load_activity(cfg: ActivityConfig, trace_ids: np.ndarray | None = None) -> Tensor:
    """Load activity traces from a zarr v3 array on disk.

    Opens ``cfg.traces_path`` via tensorstore (zarr3 driver), reads the full
    array into a CPU float32 tensor of shape ``(T, N)`` and clips values to
    ``[cfg.min_value, cfg.max_value]``.

    Args:
        cfg: Activity configuration with path and clamp values.
        trace_ids: Optional array of column indices (ZB_IDs) to select a
            subset of neurons. When None, all columns are returned.
    """
    arr = read_array(cfg.traces_path)
    traces = torch.from_numpy(arr)
    traces = traces.clamp(min=cfg.min_value, max=cfg.max_value)
    if trace_ids is not None:
        return traces[:, trace_ids]
    return traces
