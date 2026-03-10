#!/usr/bin/env python3
"""Download neuron soma info and connectome from neuprint.

Writes parquet files to a timestamped directory and updates a `latest` symlink
on success. Directory name encodes the date/time and git SHA of the code.

Usage:
    python scripts/download_neuprint.py

Paths are configured in src/zap_model/local_paths.py (see local_paths.example.py):
    NEUPRINT_DOWNLOAD_DIR

Output structure:
    <output_root>/
    ├── 20260309_143022_a1b2c3d/
    │   ├── soma.parquet
    │   └── connections.parquet
    └── latest -> 20260309_143022_a1b2c3d/
"""

import subprocess
from datetime import datetime

from fishfuncem.em.NeuprintServer import NeuprintServer

from zap_model.data.em import fetch_connections, fetch_soma_info
from zap_model.local_paths import NEUPRINT_DOWNLOAD_DIR


def _git_sha() -> str:
    """Return short git SHA, with -dirty suffix if working tree is not clean."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet"],
        capture_output=True,
        check=False,
    )
    if dirty.returncode != 0:
        sha += "-dirty"
    return sha


def main():
    output_root = NEUPRINT_DOWNLOAD_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = _git_sha()
    dirname = f"{timestamp}_{sha}"
    output_dir = output_root / dirname

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}", flush=True)

    server = NeuprintServer()
    client = server.client

    print("Fetching soma info ...", flush=True)
    soma_df = fetch_soma_info(client)
    soma_path = output_dir / "soma.parquet"
    soma_df.write_parquet(soma_path)
    print(f"  {soma_df.shape[0]:,} neurons -> {soma_path}", flush=True)

    print("Fetching connections ...", flush=True)
    conn_df = fetch_connections(client, soma_df)
    conn_path = output_dir / "connections.parquet"
    conn_df.write_parquet(conn_path)
    print(f"  {conn_df.shape[0]:,} connections -> {conn_path}", flush=True)

    # Update latest symlink
    latest = output_root / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(dirname)
    print(f"  latest -> {dirname}", flush=True)


if __name__ == "__main__":
    main()
