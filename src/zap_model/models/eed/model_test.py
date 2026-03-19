"""EED model tests."""

from __future__ import annotations

import unittest

import torch

from zap_model.models.eed.config import EEDModelConfig
from zap_model.models.eed.model import EEDModel
from zap_model.nn.mlp import MLPConfig
from zap_model.training.test_util import assert_trainable

_TINY_CFG = EEDModelConfig(
    latent_dim=2,
    codec=MLPConfig(hidden_dims=(8,)),
    evolver=MLPConfig(hidden_dims=(8,), activation="Tanh"),
    rollout_steps=3,
)


class TestEEDModel(unittest.TestCase):
    def test_satisfies_trainable(self) -> None:
        model = EEDModel(num_neurons=4, cfg=_TINY_CFG)
        batch = torch.randn(2, 4, 4)  # (B=2, T=4, N=4)
        assert_trainable(model, batch)


if __name__ == "__main__":
    unittest.main()
