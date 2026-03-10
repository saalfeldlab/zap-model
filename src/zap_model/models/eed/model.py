"""Encoder-evolver-decoder model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from zap_model.models.eed.config import EEDModelConfig  # noqa: TC001 — used at runtime
from zap_model.nn.mlp import MLP


@dataclass
class EEDLosses:
    """Losses returned by :meth:`EEDModel.training_step`."""

    total: Tensor


class EEDModel(nn.Module):
    """Simplified encoder-evolver-decoder for neural activity prediction.

    The evolver is residual (``z + evolver(z)``) with zero-initialized final layer so
    that the model starts as an autoencoder.
    """

    def __init__(self, num_neurons: int, cfg: EEDModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MLP(num_neurons, cfg.latent_dim, cfg.encoder)
        self.decoder = MLP(cfg.latent_dim, num_neurons, cfg.decoder)
        self.evolver = MLP(cfg.latent_dim, cfg.latent_dim, cfg.evolver)

        # Zero-init final layer of evolver so initial behaviour is autoencoder
        final_linear: nn.Linear = self.evolver.layers[-1]
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def evolve(self, z: Tensor) -> Tensor:
        return z + self.evolver(z)

    def forward(self, x: Tensor) -> Tensor:
        """Single step: encode → evolve → decode."""
        return self.decode(self.evolve(self.encode(x)))

    @torch.compile(mode="reduce-overhead")
    def training_step(self, batch: Tensor) -> EEDLosses:
        """Compute rollout loss over ``cfg.rollout_steps`` time steps.

        Args:
            batch: Tensor of shape ``(B, T, N)`` where ``T >= rollout_steps + 1``.

        Returns:
            :class:`EEDLosses` with mean MSE across rollout steps.
        """
        z = self.encode(batch[:, 0, :])
        total_mse = torch.zeros((), device=batch.device)
        for t in range(self.cfg.rollout_steps):
            x_hat = self.decode(z)
            total_mse = total_mse + F.mse_loss(x_hat, batch[:, t + 1, :])
            z = self.evolve(z)
        total_mse = total_mse / self.cfg.rollout_steps
        return EEDLosses(total=total_mse)

    def param_groups(self) -> list[dict]:
        """Return optimizer parameter groups."""
        return [{"params": list(self.parameters())}]
