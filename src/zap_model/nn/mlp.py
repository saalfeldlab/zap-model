"""Simple MLP building block."""

from __future__ import annotations

from pydantic import BaseModel
from torch import Tensor, nn


class MLPConfig(BaseModel, frozen=True, extra="forbid"):
    """Configuration for an MLP.

    Attributes:
        hidden_dims: Width of each hidden layer.
        activation: Name of a ``torch.nn`` activation class (e.g. "ReLU", "Tanh").
    """

    hidden_dims: tuple[int, ...]
    activation: str = "ReLU"


class MLP(nn.Module):
    """Multi-layer perceptron: Linear → Activation → … → Linear."""

    def __init__(self, in_dim: int, out_dim: int, cfg: MLPConfig) -> None:
        super().__init__()
        activation_cls = getattr(nn, cfg.activation)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation_cls())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
