"""EM data: neuron soma metadata and connectome from neuprint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

import polars as pl
from fishfuncem.utils.coords import voxel_to_nm


class SomaCol(StrEnum):
    """Column names for the neuron soma dataframe."""

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
