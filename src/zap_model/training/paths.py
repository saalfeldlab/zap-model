"""Run directory creation for training experiments."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime via TRAINING_DIR fallback

try:
    from zap_model.local_paths import TRAINING_DIR
except ImportError:
    TRAINING_DIR = None  # type: ignore[assignment]


def git_sha() -> str:
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


def create_run_dir(expt_name: str) -> Path:
    """Create and return ``TRAINING_DIR / <expt_name>_<date>_<commit>/``."""
    if TRAINING_DIR is None:
        msg = "TRAINING_DIR not configured — create src/zap_model/local_paths.py"
        raise RuntimeError(msg)
    training_dir = TRAINING_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = git_sha()
    dirname = f"{expt_name}_{timestamp}_{sha}"
    run_dir = training_dir / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
