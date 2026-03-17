"""Shared activity data loading — disk → CPU tensor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import tensorstore as ts
import torch
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np

    from zap_model.data.config import ActivityConfig

# Minimum and maximum values for dF/F in zapbench.
MIN_MAX_VALUES = (-0.25, 1.5)


@dataclass
class ActivityData:
    """Activity traces loaded into CPU memory.

    Attributes:
        traces: (T, N) float32 tensor on CPU.
        num_frames: T — total number of frames across all conditions.
        num_neurons: N — number of neurons.
    """

    traces: Tensor
    num_frames: int
    num_neurons: int


def load_activity(cfg: ActivityConfig) -> ActivityData:
    """Load activity traces from a zarr v3 array on disk.

    Opens ``cfg.traces_path`` via tensorstore (zarr3 driver), reads the full
    array into a CPU float32 tensor, and clips values to
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
    traces = traces.clamp(min=cfg.min_value, max=cfg.max_value)
    num_frames, num_neurons = traces.shape
    return ActivityData(traces=traces, num_frames=num_frames, num_neurons=num_neurons)
