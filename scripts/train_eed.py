#!/usr/bin/env python
"""CLI entrypoint for EED model training.

Usage::

    python scripts/train_eed.py <expt_name> [--help] [--training.batch-size N] ...
"""

from __future__ import annotations

# Torch 2.10's NVIDIA libs (e.g. libcublas) use RUNPATH which only searches their own dirs,
# falling back to the system libstdc++ (missing CXXABI_1.3.15). Importing tensorstore first
# loads the conda libstdc++ via its RPATH ($ORIGIN/../../..), and it stays cached process-wide.
import logging
import re
import sys

import tensorstore  # noqa: F401 -- must be imported before torch to load conda's libstdc++ first.
import tyro

from zap_model.data.activity import load_activity
from zap_model.data.em import SomaCol, resolve_neurons
from zap_model.models.eed.config import EEDConfig
from zap_model.models.eed.data import make_eed_data
from zap_model.models.eed.model import EEDModel
from zap_model.training.paths import create_run_dir
from zap_model.training.trainer import train
from zap_model.training.util import get_device, seed_everything


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print(
            "usage: python scripts/train_eed.py <expt_name> [overrides...]\n"
            "  expt_name must match [A-Za-z0-9_]+",
            file=sys.stderr,
        )
        sys.exit(1)

    expt_name = sys.argv[1]
    if not re.fullmatch(r"[A-Za-z0-9_]+", expt_name):
        print(f"error: expt_name must match [A-Za-z0-9_]+, got: {expt_name}", file=sys.stderr)
        sys.exit(1)

    cfg = tyro.cli(EEDConfig, args=sys.argv[2:])
    seed_everything(cfg.training.seed)

    run_dir = create_run_dir(expt_name)
    print(f"run dir: {run_dir}", flush=True)

    if cfg.data.neuron_ids_path is not None:
        neuron_mapping = resolve_neurons(
            cfg.data.neuprint,
            cfg.data.neuron_ids_path,
            restrict_zb_ids=True,
        )
        trace_ids = neuron_mapping[SomaCol.ZB_ID].to_numpy()
    else:
        trace_ids = None

    traces = load_activity(cfg.data.activity, trace_ids=trace_ids)
    device = get_device()
    data = make_eed_data(
        traces,
        cfg.data.splits,
        cfg.model,
        cfg.training.batch_size,
        device,
    )

    batches_per_full_pass = data.num_train_windows // cfg.training.batch_size
    print(
        f"training windows: {data.num_train_windows:,}, "
        f"batches for one full pass: {batches_per_full_pass}, "
        f"batches_per_epoch: {cfg.training.batches_per_epoch}",
        flush=True,
    )
    print(
        f"validation windows: {data.num_val_windows:,}, "
        f"val_rollout_steps: {cfg.model.val_rollout_steps}",
        flush=True,
    )

    (run_dir / "command_line.txt").write_text("\n".join(sys.argv))

    num_neurons = traces.shape[1]
    model = EEDModel(num_neurons=num_neurons, cfg=cfg.model)
    train(model, data.train_iter, data.val_iter, cfg.training, run_dir)


if __name__ == "__main__":
    main()
