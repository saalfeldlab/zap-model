"""EED model and run configuration."""

from __future__ import annotations

from pydantic import BaseModel

from zap_model.data.config import DataConfig  # noqa: TC001 — pydantic needs at runtime
from zap_model.nn.mlp import MLPConfig
from zap_model.training.trainer import TrainingConfig  # noqa: TC001 — pydantic needs at runtime


class EEDModelConfig(BaseModel, extra="forbid"):
    """Configuration for the encoder-evolver-decoder model."""

    latent_dim: int = 64
    codec: MLPConfig = MLPConfig(hidden_dims=(128, 128))
    evolver: MLPConfig = MLPConfig(hidden_dims=(128, 128), activation="Tanh")
    rollout_steps: int = 10


class EEDConfig(BaseModel, extra="forbid"):
    """Top-level run configuration for EED training."""

    data: DataConfig
    training: TrainingConfig
    model: EEDModelConfig = EEDModelConfig()
