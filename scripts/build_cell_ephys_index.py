#!/usr/bin/env python
"""Build cell_ephys_index zarr from raw ephys, segmentation, and flow fields.

For each cell and timepoint, computes the ephys sample index by:
1. Computing imaging_sample_index from TTL peaks in the raw ephys binary
2. Computing cell centroids from the segmentation volume
3. Correcting each cell's z-coordinate per timepoint using flow fields
4. Indexing: cell_ephys_index[t, cell] = imaging_sample_index[t, corrected_z]

Usage:
    python scripts/build_cell_ephys_index.py

Paths are configured in src/zap_model/local_paths.py (see local_paths.example.py):
    ZAPBENCH_LOCAL_PATH, ZAPBENCH_GCS_URI
"""

import json
import time

import numpy as np
import tensorstore as ts
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from tqdm import tqdm

from zap_model.data.ephys import NUM_FRAMES, ZAP_CELL_EPHYS_INDEX_PATH, EphysChannel, load_raw
from zap_model.local_paths import ZAPBENCH_GCS_URI, ZAPBENCH_LOCAL_PATH

# Flow field grid strides (aligned-space pixels per grid point)
STRIDE_X = 16
STRIDE_Y = 16
STRIDE_Z = 2

# Zarr chunk size
CHUNK_SIZE = 512

# GCS retry parameters
MAX_RETRIES = 5
TIMEOUT_S = 30


# TTL peak detection thresholds
_TTL_HIGH_DISTANCE = 500
_TTL_HIGH_HEIGHT = 3.55
_TTL_LOW_DISTANCE = 50
_TTL_LOW_HEIGHT = 1.0
_TTL_MERGE_RADIUS = 5


def _compute_imaging_sample_index(ttl: np.ndarray, num_z_slices: int) -> np.ndarray:
    """Map each imaging frame to its ephys sample index.

    Returns (NUM_TIMEPOINTS, num_z_slices) int32 array where entry [t, z]
    is the ephys sample index when z-plane z was imaged at timepoint t.

    High TTL peaks mark volume boundaries. Low peaks (num_z_slices per volume)
    mark individual z-planes. The last ~8 timepoints are linearly extrapolated
    because the recording is truncated mid-volume.
    """
    high_peaks, _ = find_peaks(ttl, distance=_TTL_HIGH_DISTANCE, height=_TTL_HIGH_HEIGHT)
    low_all, _ = find_peaks(ttl, distance=_TTL_LOW_DISTANCE, height=_TTL_LOW_HEIGHT)

    # Remove low peaks that coincide with high peaks
    tree = KDTree(low_all.reshape(-1, 1))
    near = tree.query_ball_point(high_peaks.reshape(-1, 1), r=_TTL_MERGE_RADIUS)
    keep = np.ones(len(low_all), dtype=bool)
    for group in near:
        for j in group:
            keep[j] = False
    low_peaks = low_all[keep]

    # Each complete volume should have exactly num_z_slices low peaks
    num_complete = len(high_peaks) - 1
    expected_low_peaks = num_complete * num_z_slices
    assert len(low_peaks) >= expected_low_peaks, (
        f"Expected at least {expected_low_peaks} low peaks "
        f"({num_complete} volumes x {num_z_slices} z-slices), got {len(low_peaks)}"
    )

    # Place known z-plane sample indices
    result = np.full((NUM_FRAMES, num_z_slices), -1, dtype=np.int32)
    result.ravel()[: len(low_peaks)] = low_peaks

    # Linearly extrapolate the truncated tail
    delta = np.diff(result[:num_complete], axis=0).mean(axis=0)
    last = result[num_complete - 1]
    for i in range(NUM_FRAMES - num_complete):
        row = num_complete + i
        extrap = np.round(last + (i + 1) * delta).astype(np.int32)
        result[row] = np.where(result[row] >= 0, result[row], extrap)

    assert (result >= 0).all(), "Negative indices remain after extrapolation"
    return result


def _read_with_retry(store, label=""):
    """Read from tensorstore with retry for GCS timeouts."""
    for attempt in range(MAX_RETRIES):
        try:
            return store.read().result(timeout=TIMEOUT_S)
        except TimeoutError:
            print(f"  {label} timeout (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(2**attempt)
    msg = f"{label} failed after {MAX_RETRIES} attempts"
    raise TimeoutError(msg)


def _load_segmentation(data_root: str) -> np.ndarray:
    """Load segmentation volume from local filesystem."""
    seg_path = f"{data_root}/segmentation"
    print(f"Loading segmentation from {seg_path} ...", flush=True)
    ds = ts.open({"open": True, "driver": "zarr3", "kvstore": seg_path}).result()
    seg = ds.read().result()
    print(f"  shape: {seg.shape}, dtype: {seg.dtype}", flush=True)
    return np.asarray(seg)


def _compute_cell_centroids(segmentation: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute (num_cells, 3) centroids in (x, y, z) order from labeled volume."""
    print("Computing cell centroids ...", flush=True)
    xi, yi, zi = np.where(segmentation > 0)
    cell_ids = segmentation[xi, yi, zi].astype(np.int64) - 1  # 0-indexed
    num_cells = int(cell_ids.max() + 1)

    counts = np.bincount(cell_ids, minlength=num_cells).astype(np.float64)
    centroids = np.zeros((num_cells, 3), dtype=np.float64)
    w = [coord.astype(np.float64) for coord in (xi, yi, zi)]
    for axis, wi in enumerate(w):
        centroids[:, axis] = np.bincount(cell_ids, weights=wi, minlength=num_cells) / counts

    zmin, zmax = centroids[:, 2].min(), centroids[:, 2].max()
    print(f"  {num_cells:,} cells, z range [{zmin:.1f}, {zmax:.1f}]", flush=True)
    return centroids, num_cells


def _open_flow_fields(gcs_uri: str) -> tuple[ts.TensorStore, int]:
    """Open flow fields from GCS and return (store, time_chunk_size)."""
    print(f"Opening flow fields from {gcs_uri}/flow_fields ...", flush=True)
    ds = ts.open({"open": True, "driver": "zarr3", "kvstore": f"{gcs_uri}/flow_fields"}).result()

    kvstore = ts.KvStore.open(f"{gcs_uri}/flow_fields/").result()
    meta = json.loads(kvstore.read("zarr.json").result().value)
    time_chunk = meta["chunk_grid"]["configuration"]["chunk_shape"][-1]

    print(f"  shape: {ds.shape}, time chunk: {time_chunk}", flush=True)
    return ds, time_chunk


def main():
    data_root = str(ZAPBENCH_LOCAL_PATH)
    gcs_uri = ZAPBENCH_GCS_URI
    output_path = str(ZAP_CELL_EPHYS_INDEX_PATH)
    raw_ephys_path = str(ZAPBENCH_LOCAL_PATH / "stimuli_raw" / "stimuli_and_ephys.10chFlt")

    # 1. Load segmentation and compute cell centroids
    segmentation = _load_segmentation(data_root)
    num_z_slices = segmentation.shape[2]
    centroids, num_cells = _compute_cell_centroids(segmentation)
    del segmentation

    # 2. Compute imaging_sample_index from raw ephys TTL
    print(f"Loading raw ephys from {raw_ephys_path}", flush=True)
    raw = load_raw(raw_ephys_path)
    print(f"  shape: {raw.shape}", flush=True)

    print("Computing imaging_sample_index from TTL ...", flush=True)
    ttl = raw[:, EphysChannel.TTL]
    imaging_sample_index = _compute_imaging_sample_index(ttl, num_z_slices)
    print(f"  shape: {imaging_sample_index.shape}", flush=True)

    num_timepoints = imaging_sample_index.shape[0]

    # Precompute flow-field grid coordinates for all cells: (z/Sz, y/Sy, x/Sx)
    grid_coords = np.array(
        [
            centroids[:, 2] / STRIDE_Z,
            centroids[:, 1] / STRIDE_Y,
            centroids[:, 0] / STRIDE_X,
        ]
    )

    # 3. Open flow fields
    ds_flow, time_chunk = _open_flow_fields(gcs_uri)

    # 4. Build cell_ephys_index
    cell_ephys_index = np.empty((num_timepoints, num_cells), dtype=np.int32)
    num_chunks = (num_timepoints + time_chunk - 1) // time_chunk

    for ci in tqdm(range(num_chunks), desc="flow field chunks"):
        t_start = ci * time_chunk
        t_end = min(t_start + time_chunk, num_timepoints)

        flow_chunk = _read_with_retry(ds_flow[:, :, :, :, t_start:t_end], f"flow chunk {ci}")

        for i, t in enumerate(range(t_start, t_end)):
            z_offset = map_coordinates(
                flow_chunk[2, :, :, :, i],
                grid_coords,
                order=3,
                mode="nearest",
            )
            corrected_z = centroids[:, 2] + z_offset / 4.0
            corrected_z_int = np.round(corrected_z).astype(np.int32)
            cell_ephys_index[t, :] = imaging_sample_index[t, corrected_z_int]

    # 5. Write to zarr
    print(f"Writing {output_path} [{num_timepoints}, {num_cells}] ...", flush=True)
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": output_path},
        "metadata": {
            "shape": [num_timepoints, num_cells],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [CHUNK_SIZE, CHUNK_SIZE]},
            },
            "chunk_key_encoding": {"name": "default"},
            "data_type": "int32",
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5}},
            ],
        },
        "create": True,
        "delete_existing": True,
    }
    store = ts.open(spec).result()
    store.write(cell_ephys_index).result()
    print(f"  sample index range: [{cell_ephys_index.min()}, {cell_ephys_index.max()}]", flush=True)


if __name__ == "__main__":
    main()
