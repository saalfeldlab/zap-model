"""CLI entrypoint for EED model training.

Usage::

    python scripts/train_eed.py --help
"""

from __future__ import annotations

import tyro

from zap_model.data.activity import load_activity
from zap_model.models.eed.config import EEDConfig
from zap_model.models.eed.data import make_eed_data
from zap_model.models.eed.model import EEDModel
from zap_model.training.trainer import train
from zap_model.training.util import get_device, seed_everything


def main() -> None:
    cfg = tyro.cli(EEDConfig)
    seed_everything(cfg.training.seed)

    activity = load_activity(cfg.data.activity)
    device = get_device()
    data = make_eed_data(
        activity,
        cfg.data.splits,
        cfg.model,
        cfg.training.batch_size,
        device,
    )

    # Compute and set batches_per_epoch from data
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={
                    "batches_per_epoch": data.num_train_windows // cfg.training.batch_size,
                }
            ),
        }
    )

    model = EEDModel(num_neurons=activity.num_neurons, cfg=cfg.model)
    train(model, data.train_iter, data.val_iter, cfg.training)


if __name__ == "__main__":
    main()
