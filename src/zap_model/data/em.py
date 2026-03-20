"""EM data: neuron soma metadata and connectome from neuprint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse
from fishfuncem.utils.coords import voxel_to_nm

if TYPE_CHECKING:
    from pathlib import Path

    from zap_model.data.config import NeuprintConfig


class SomaCol(StrEnum):
    """Column names for the neuron soma dataframe.

    This is an alias for the neuprint column names that are too verbose.
    """

    ID = "id"  # n.bodyId
    TYPE = "type"  # n.type
    ZB_ID = "zb_id"  # n.zapbenchId (cast to int32, 0-indexed)
    STATUS = "status"  # n.statusLabel
    X_UM = "x_um"  # n.somaLocation.x (voxel -> µm)
    Y_UM = "y_um"  # n.somaLocation.y (voxel -> µm)
    Z_UM = "z_um"  # n.somaLocation.z (voxel -> µm)


class ConnCol(StrEnum):
    """Column names for the connections dataframe."""

    ID_PRE = "id_pre"  # pre.bodyId
    WEIGHT = "weight"  # c.weight
    ID_POST = "id_post"  # post.bodyId


def fetch_soma_info(client) -> pl.DataFrame:
    """Query neuprint for per-neuron soma metadata.

    Returns a polars DataFrame with columns defined by SomaCol.
    Soma coordinates are converted from voxels to microns.
    """
    raw = client.fetch_custom(
        """
        MATCH (n:Neuron)
        RETURN n.bodyId AS id, n.type AS type, n.zapbenchId AS zb_id,
            n.statusLabel AS status,
            n.somaLocation.x AS soma_x,
            n.somaLocation.y AS soma_y,
            n.somaLocation.z AS soma_z
        """,
        format="pandas",
    )
    df = pl.from_pandas(raw).with_columns(
        pl.col(SomaCol.ZB_ID).cast(pl.Int32) - 1,
    )

    # Convert soma voxel coordinates to microns
    coords_vox = df.select(["soma_x", "soma_y", "soma_z"]).to_numpy()
    coords_um = voxel_to_nm(coords_vox) / 1000.0
    coords_df = pl.DataFrame(
        coords_um,
        schema=[SomaCol.X_UM, SomaCol.Y_UM, SomaCol.Z_UM],
    )

    return pl.concat(
        [df.drop(["soma_x", "soma_y", "soma_z"]), coords_df],
        how="horizontal",
    )


def fetch_connections(
    client,
    soma_df: pl.DataFrame,
    *,
    weight_thresh: int = 1,
    batch_size: int = 1000,
    max_workers: int = 4,
) -> pl.DataFrame:
    """Query neuprint for all connections to neurons in soma_df.

    Returns a polars DataFrame with columns defined by ConnCol.
    Queries are batched by post-synaptic neuron ID and run in parallel.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, batch in soma_df.with_row_index().group_by(
            [pl.col("index") // batch_size],
            maintain_order=True,
        ):
            ids = batch[SomaCol.ID].to_list()
            q = f"""
                MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
                WHERE c.weight >= {weight_thresh} AND post.bodyId IN {ids}
                RETURN pre.bodyId AS {ConnCol.ID_PRE},
                    c.weight AS {ConnCol.WEIGHT},
                    post.bodyId AS {ConnCol.ID_POST}
            """
            results.append(executor.submit(client.fetch_custom, q))

    return pl.concat([pl.from_dataframe(r.result()) for r in results])


@dataclass
class Connectivity:
    """Sparse connectivity matrix with neuron ID mapping.

    Attributes:
        W: (N, N) sparse CSC matrix. W[i, j] = synapse count from neuron i to neuron j.
            CSC layout is optimal for computing post-synaptic input: ``W.T @ activity``
            is a CSR @ dense operation on the transpose view.
        body_ids: (N,) array of neuprint body IDs, one per matrix row/col.
    """

    W: scipy.sparse.csc_matrix
    body_ids: np.ndarray

    def save(self, path: Path) -> None:
        """Save to a zarr3 directory.

        Layout::

            path/
                pre/          (nnz,) int32 — pre-synaptic matrix indices
                post/         (nnz,) int32 — post-synaptic matrix indices
                weight/       (nnz,) float32 — synapse counts
                neuron_ids/   (N,) int64 — body IDs
                metadata.json {\"num_neurons\": N}
        """
        import json

        from zap_model.data.zarr import write_array

        path.mkdir(parents=True, exist_ok=True)

        coo = self.W.tocoo()
        write_array(path / "pre", coo.row.astype(np.int32))
        write_array(path / "post", coo.col.astype(np.int32))
        write_array(path / "weight", coo.data.astype(np.float32))
        write_array(path / "neuron_ids", self.body_ids.astype(np.int64))

        metadata = {"num_neurons": self.W.shape[0]}
        (path / "metadata.json").write_text(json.dumps(metadata))

    @classmethod
    def load(cls, path: Path) -> Connectivity:
        """Load from a zarr3 directory written by :meth:`save`."""
        import json

        from zap_model.data.zarr import read_array

        pre = read_array(path / "pre")
        post = read_array(path / "post")
        weight = read_array(path / "weight")
        neuron_ids = read_array(path / "neuron_ids")

        n = json.loads((path / "metadata.json").read_text())["num_neurons"]

        W = scipy.sparse.csc_matrix(
            (weight.astype(np.float32), (pre.astype(np.int32), post.astype(np.int32))),
            shape=(n, n),
        )
        return cls(W=W, body_ids=neuron_ids)


def resolve_neurons(
    neuprint_cfg: NeuprintConfig,
    body_ids_path: Path | None = None,
    restrict_zb_ids: bool = False,
) -> pl.DataFrame:
    """Resolve the set of neurons to use, returning a DataFrame with ID and ZB_ID columns.

    Args:
        neuprint_cfg: Neuprint configuration specifying ``data_dir`` and ``status_filter``.
        body_ids_path: Optional parquet file with an ``id`` column listing body IDs
            to include. When None, all neurons passing the status filter are used.
        restrict_zb_ids: When True, further filter to neurons with a valid zapbench ID.

    Returns:
        A DataFrame with at least ``SomaCol.ID`` and ``SomaCol.ZB_ID`` columns.
    """
    soma_df = pl.read_parquet(neuprint_cfg.data_dir / "soma.parquet")

    if body_ids_path is not None:
        ids_df = pl.read_parquet(body_ids_path)
        valid_ids = soma_df.select(SomaCol.ID, SomaCol.ZB_ID)
        soma_df = ids_df.select(SomaCol.ID).join(valid_ids, on=SomaCol.ID, how="left")
    else:
        soma_df = soma_df.filter(pl.col(SomaCol.STATUS).is_in(neuprint_cfg.status_filter))

    if restrict_zb_ids:
        soma_df = soma_df.filter(pl.col(SomaCol.ZB_ID).is_not_null())

    return soma_df.select(SomaCol.ID, SomaCol.ZB_ID)


def build_connectivity(
    neuprint_cfg: NeuprintConfig,
    body_ids: np.ndarray,
) -> Connectivity:
    """Build a sparse connectivity matrix from downloaded neuprint parquet files.

    Args:
        neuprint_cfg: Neuprint configuration specifying ``data_dir`` and ``min_weight``.
        body_ids: Array of body IDs defining the neuron set and matrix ordering.

    Returns:
        A :class:`Connectivity` with the sparse weight matrix and body ID array.
    """
    conn_df = pl.read_parquet(neuprint_cfg.data_dir / "connections.parquet")

    n = len(body_ids)
    id_to_idx = {bid: idx for idx, bid in enumerate(body_ids)}

    # Filter connections: both endpoints in neuron set, min weight
    id_set = set(id_to_idx)
    conn_df = conn_df.filter(
        pl.col(ConnCol.ID_PRE).is_in(id_set)
        & pl.col(ConnCol.ID_POST).is_in(id_set)
        & (pl.col(ConnCol.WEIGHT) >= neuprint_cfg.min_weight)
    )

    # Map body IDs to matrix indices
    pre_ids = conn_df[ConnCol.ID_PRE].to_numpy()
    post_ids = conn_df[ConnCol.ID_POST].to_numpy()
    weights = conn_df[ConnCol.WEIGHT].to_numpy()

    pre_idx = np.array([id_to_idx[x] for x in pre_ids], dtype=np.int32)
    post_idx = np.array([id_to_idx[x] for x in post_ids], dtype=np.int32)

    W = scipy.sparse.csc_matrix(
        (weights.astype(np.float32), (pre_idx, post_idx)),
        shape=(n, n),
    )

    return Connectivity(W=W, body_ids=body_ids)
