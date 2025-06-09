import unittest
from types import SimpleNamespace

import torch

from splatnlp.dashboard.app import DASHBOARD_CONTEXT
from splatnlp.dashboard.components.ablation_component import (
    _compute_feature_activation,
)
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder


class TestAblationComponent(unittest.TestCase):
    def setUp(self):
        DASHBOARD_CONTEXT.vocab = {"<PAD>": 0, "a": 1, "b": 2}
        DASHBOARD_CONTEXT.weapon_vocab = {"w": 0}
        DASHBOARD_CONTEXT.device = "cpu"

        self.model = SetCompletionModel(
            vocab_size=3,
            weapon_vocab_size=1,
            embedding_dim=4,
            hidden_dim=4,
            output_dim=3,
            num_layers=1,
            num_heads=1,
            num_inducing_points=1,
            use_layer_norm=False,
            dropout=0.0,
            pad_token_id=0,
        )
        self.sae = SparseAutoencoder(input_dim=4, expansion_factor=2)
        DASHBOARD_CONTEXT.primary_model = self.model
        DASHBOARD_CONTEXT.sae_model = self.sae

    def tearDown(self):
        DASHBOARD_CONTEXT.__dict__.clear()

    def test_compute_feature_activation_returns_float(self):
        act = _compute_feature_activation(["a", "b"], "w", 0)
        self.assertIsInstance(act, float)


if __name__ == "__main__":
    unittest.main()
