"""Zarr3 read/write helpers via tensorstore."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import tensorstore as ts

if TYPE_CHECKING:
    import numpy as np

# Map numpy dtype names to zarr3 data_type strings.
_DTYPE_MAP = {
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
    "float64": "float64",
}


def write_array(
    path: Path,
    arr: np.ndarray,
    chunk_shape: list[int] | None = None,
) -> None:
    """Write a numpy array as a zarr3 array."""
    dtype_str = _DTYPE_MAP.get(arr.dtype.name)
    if dtype_str is None:
        msg = f"Unsupported dtype {arr.dtype} — add it to _DTYPE_MAP"
        raise ValueError(msg)

    if chunk_shape is None:
        chunk_shape = list(arr.shape)

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(Path(path).resolve())},
        "metadata": {
            "shape": list(arr.shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": chunk_shape},
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
        "kvstore": {"driver": "file", "path": str(Path(path).resolve())},
    }
    store = ts.open(spec, read=True).result()
    return store.read().result()


def open_gcs_zarr(gcs_uri: str) -> ts.TensorStore:
    """Open a zarr3 array on GCS for reading."""
    return ts.open({"open": True, "driver": "zarr3", "kvstore": gcs_uri}).result()


def gcs_chunk_shape(gcs_uri: str) -> list[int]:
    """Read the chunk shape from a zarr3 array's metadata on GCS."""
    kvstore = ts.KvStore.open(f"{gcs_uri}/").result()
    meta = json.loads(kvstore.read("zarr.json").result().value)
    return meta["chunk_grid"]["configuration"]["chunk_shape"]


def read_with_retry(
    store: ts.TensorStore,
    *,
    max_retries: int = 5,
    timeout_s: int = 30,
    label: str = "",
) -> np.ndarray:
    """Read from a tensorstore with exponential backoff on timeout."""
    for attempt in range(max_retries):
        try:
            return store.read().result(timeout=timeout_s)
        except TimeoutError:
            print(f"  {label} timeout (attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(2**attempt)
    msg = f"{label} failed after {max_retries} attempts"
    raise TimeoutError(msg)
