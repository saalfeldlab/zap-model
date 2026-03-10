"""Shared interface checker for trainable models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn


def assert_trainable(model: nn.Module, batch: Tensor) -> None:
    """Verify that *model* satisfies the trainable interface expected by ``train()``.

    Checks:
    - ``model.param_groups()`` returns a list of dicts containing ``"params"``.
    - ``model.training_step(batch)`` returns a dataclass with a scalar ``total`` field.
    """
    groups = model.param_groups()
    assert isinstance(groups, list) and len(groups) > 0
    assert "params" in groups[0]

    losses = model.training_step(batch)
    assert dataclasses.is_dataclass(losses)
    assert hasattr(losses, "total")
    for f in dataclasses.fields(losses):
        val = getattr(losses, f.name)
        assert val.shape == ()  # scalar
