"""Loss accumulation utilities."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class LossAccumulator:
    """Accumulates per-field losses from dataclass instances and computes running means.

    Expects each input to be a dataclass whose fields are scalar :class:`~torch.Tensor` values.
    """

    _sums: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def accumulate(self, losses: object) -> None:
        """Add a batch of losses. Values are detached and converted to Python floats."""
        for f in dataclasses.fields(losses):
            val: Tensor = getattr(losses, f.name)
            self._sums[f.name] = self._sums.get(f.name, 0.0) + val.detach().item()
            self._counts[f.name] = self._counts.get(f.name, 0) + 1

    def mean(self) -> dict[str, float]:
        """Return the mean of each accumulated field."""
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def __getitem__(self, key: str) -> float:
        return self._sums[key] / self._counts[key]

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()
