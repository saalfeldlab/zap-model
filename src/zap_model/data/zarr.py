"""Zarr3 read/write helpers via tensorstore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tensorstore as ts

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

# Map numpy dtype names to zarr3 data_type strings.
_DTYPE_MAP = {
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
    "float64": "float64",
}


def write_array(path: Path, arr: np.ndarray) -> None:
    """Write a numpy array as a zarr3 array."""
    dtype_str = _DTYPE_MAP.get(arr.dtype.name)
    if dtype_str is None:
        msg = f"Unsupported dtype {arr.dtype} — add it to _DTYPE_MAP"
        raise ValueError(msg)

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(arr.shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(arr.shape)},
            },
            "chunk_key_encoding": {"name": "default"},
            "data_type": dtype_str,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5}},
            ],
        },
        "create": True,
        "delete_existing": True,
    }
    store = ts.open(spec).result()
    store.write(arr).result()


def read_array(path: Path) -> np.ndarray:
    """Read a zarr3 array into a numpy array."""
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
    }
    store = ts.open(spec, read=True).result()
    return store.read().result()
