"""Shared activity data loading — disk → CPU tensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tensorstore as ts
import torch
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np

    from zap_model.data.config import ActivityConfig


def load_activity(cfg: ActivityConfig) -> Tensor:
    """Load activity traces from a zarr v3 array on disk.

    Opens ``cfg.traces_path`` via tensorstore (zarr3 driver), reads the full
    array into a CPU float32 tensor of shape ``(T, N)`` and clips values to
    ``[cfg.min_value, cfg.max_value]``.
    """
    spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(cfg.traces_path),
        },
    }
    store = ts.open(spec, read=True).result()
    arr: np.ndarray = store.read().result()
    traces = torch.tensor(arr, dtype=torch.float32)
    return traces.clamp(min=cfg.min_value, max=cfg.max_value)
