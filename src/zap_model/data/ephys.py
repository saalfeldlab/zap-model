"""Ephys data loading.

Glossary
--------
Frame : Nominal timestep index (0..T-1) — one row of the traces matrix.
        All cells share the same Frame index, but were not acquired simultaneously.
AcqIndex : Index into the 6 kHz ephys acquisition stream. Represents the actual
           wall-clock time at which a specific cell was imaged. Different cells in
           the same Frame have different AcqIndex values because the microscope
           scans z-planes sequentially.

cell_ephys_index maps (Frame, Cell) -> AcqIndex.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NewType

import numpy as np

from zap_model.local_paths import OUTPUT_DATA_DIR

AcqIndex = NewType("AcqIndex", int)

SAMPLING_FREQUENCY_HZ = 6000
NUM_RAW_CHANNELS = 10

ZAP_CELL_EPHYS_INDEX_PATH = OUTPUT_DATA_DIR / "cell_ephys_index.zarr"


class EphysChannel(IntEnum):
    """Source channel index in the 10-channel raw binary."""

    CH1 = 0
    CH2 = 1
    TTL = 2
    STIM_PARAM4 = 3
    TRIAL_ID = 4
    STIM_PARAM3 = 6
    VISUAL_VELOCITY = 8


def load_raw(path: str) -> np.ndarray[tuple[AcqIndex, int], np.dtype[np.float32]]:
    """Load raw 10-channel ephys binary -> (num_samples, 10) float32.

    Note: this recording is truncated and we will eventually use some simple
    extrapolation logic to read beyond the end of the file.
    """
    data = np.fromfile(path, dtype=np.float32)
    if data.size % NUM_RAW_CHANNELS:
        msg = f"Data size {data.size} not divisible by {NUM_RAW_CHANNELS}"
        raise ValueError(msg)
    return data.reshape(-1, NUM_RAW_CHANNELS)
