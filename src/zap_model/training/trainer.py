"""Shared training loop.

The trainer is model-agnostic. Models must implement the following duck-typed interface:

    training_step(batch: Tensor) -> dataclass
        Run a forward pass and return a dataclass of scalar losses.
        Must have a ``total`` field used for backprop. All fields are logged.
    param_groups() -> list[dict]
        Return optimizer parameter groups (list of dicts with ``"params"`` key).
    parameters(), train(), eval(), to(), state_dict()
        Standard ``nn.Module`` methods.

Compilation (e.g. ``torch.compile``) is the model's responsibility.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path  # noqa: TC003 — pydantic needs this at runtime
from typing import TYPE_CHECKING

import torch
import yaml
from pydantic import BaseModel, model_validator
from torch.utils.tensorboard import SummaryWriter

from zap_model.training.loss import LossAccumulator
from zap_model.training.util import get_device

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor, nn

log = logging.getLogger(__name__)


class TrainingConfig(BaseModel, extra="forbid"):
    """Configuration for the training loop."""

    epochs: int = 100
    batch_size: int = 32
    batches_per_epoch: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    seed: int = 42
    lr_scheduler: str | None = None
    lr_scheduler_kwargs: dict = {}
    checkpoint_every: int = 10
    log_every: int = 1

    @model_validator(mode="after")
    def _check_lr_scheduler(self) -> TrainingConfig:
        if self.lr_scheduler is not None and not hasattr(
            torch.optim.lr_scheduler, self.lr_scheduler
        ):
            msg = f"lr_scheduler={self.lr_scheduler!r} is not a class in torch.optim.lr_scheduler"
            raise ValueError(msg)
        return self


def train(
    model: nn.Module,
    train_iter: Iterator[Tensor],
    val_iter: Iterator[Tensor],
    cfg: TrainingConfig,
    run_dir: Path,
) -> Path:
    """Run the training loop and return the path to the best checkpoint.

    See module docstring for the required model interface.

    Iterators must be infinite (e.g. ``itertools.cycle`` over a shuffled source).
    Each epoch runs ``cfg.batches_per_epoch`` training steps, then one full
    validation pass of the same length.

    Seeding is the caller's responsibility — call
    :func:`~zap_model.training.util.seed_everything` before building datasets.
    """
    # use tf32 matmul
    torch.set_float32_matmul_precision("high")

    device = get_device()
    model = model.to(device)

    # Directories
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "training_config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.param_groups(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler
    scheduler = None
    if cfg.lr_scheduler is not None:
        scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler)
        scheduler = scheduler_cls(optimizer, **cfg.lr_scheduler_kwargs)

    writer = SummaryWriter(log_dir=str(tb_dir))
    best_val_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best.pt"
    global_step = 0
    t_start = time.monotonic()
    val_step_fn = getattr(model, "validation_step", model.training_step)

    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.monotonic()

        # --- Train ---
        model.train()
        step_acc = LossAccumulator()
        epoch_train_acc = LossAccumulator()
        for _ in range(cfg.batches_per_epoch):
            batch = next(train_iter)
            losses = model.training_step(batch)

            optimizer.zero_grad()
            losses.total.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            step_acc.accumulate(losses)
            epoch_train_acc.accumulate(losses)
            global_step += 1

            if cfg.log_every and global_step % cfg.log_every == 0:
                for key, val in step_acc.mean().items():
                    writer.add_scalar(f"train/{key}", val, global_step)
                step_acc.reset()

        if scheduler is not None:
            scheduler.step()

        train_total = epoch_train_acc.mean()["total"]

        # --- Validate ---
        val_batch = next(val_iter, None)
        if val_batch is not None:
            model.eval()
            val_acc = LossAccumulator()
            rollout_acc = LossAccumulator()
            t_rollout = time.monotonic()
            with torch.no_grad():
                val_batches = [val_batch]
                for _ in range(cfg.batches_per_epoch - 1):
                    vb = next(val_iter, None)
                    if vb is None:
                        break
                    val_batches.append(vb)

                for vb in val_batches:
                    val_acc.accumulate(model.training_step(vb))
                    rollout_acc.accumulate(val_step_fn(vb))
            rollout_duration = time.monotonic() - t_rollout

            val_means = val_acc.mean()
            rollout_means = rollout_acc.mean()
            for key, val in val_means.items():
                writer.add_scalar(f"val/{key}", val, global_step)
            for key, val in rollout_means.items():
                writer.add_scalar(f"val_rollout/{key}", val, global_step)

            rollout_total = rollout_means["total"]
            epoch_duration = time.monotonic() - t_epoch
            total_duration = time.monotonic() - t_start
            log.info(
                "epoch %d/%d | train: %.4e | val: %.4e | rollout: %.4e (%.1fs)"
                " | duration: %.1fs (total: %.1fs)",
                epoch,
                cfg.epochs,
                train_total,
                val_means["total"],
                rollout_total,
                rollout_duration,
                epoch_duration,
                total_duration,
            )

            # Best checkpoint (based on rollout metric)
            if rollout_total < best_val_loss:
                best_val_loss = rollout_total
                torch.save(model.state_dict(), best_ckpt_path)
        else:
            epoch_duration = time.monotonic() - t_epoch
            total_duration = time.monotonic() - t_start
            log.info(
                "epoch %d/%d | train: %.4e | (no validation data) | duration: %.1fs (total: %.1fs)",
                epoch,
                cfg.epochs,
                train_total,
                epoch_duration,
                total_duration,
            )

        # Periodic checkpoint
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

    writer.close()
    return best_ckpt_path
