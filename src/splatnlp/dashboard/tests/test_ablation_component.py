import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from dash import html

# Attempt to import components; adjust relative path if necessary for test environment
try:
    from splatnlp.dashboard.components.ablation_component import (
        display_primary_build,
        get_sae_activations,
        run_ablation_analysis,
    )
except ImportError:
    # Fallback for different execution context (e.g., running tests directly)
    # This might require specific PYTHONPATH configuration when running tests
    from ..components.ablation_component import (
        display_primary_build,
        get_sae_activations,
        run_ablation_analysis,
    )


class TestAblationComponent(unittest.TestCase):
    def setUp(self):
        # Common setup for tests, if any, can go here
        self.mock_primary_model = MagicMock()
        self.mock_sae_model = MagicMock()
        self.device = "cpu"
        self.pad_token_id = 0

        # Configure default behavior for model mocks used in get_sae_activations
        # These can be overridden in specific tests
        self.mock_primary_model.token_embed = MagicMock(
            return_value=torch.randn(1, 3, 128)
        )  # batch, seq, d_model
        self.mock_primary_model.weapon_embed = MagicMock(
            return_value=torch.randn(1, 1, 128)
        )  # batch, 1, d_model
        self.mock_primary_model.input_proj = MagicMock(
            side_effect=lambda x: x
        )  # Identity

        # Mock transformer layers - simplified: return input as is
        mock_transformer_layer = MagicMock()
        mock_transformer_layer.side_effect = lambda x, src_key_padding_mask: x
        self.mock_primary_model.transformer_layers = [mock_transformer_layer]

        self.mock_primary_model.masked_mean = MagicMock(
            return_value=torch.randn(1, 128)
        )  # batch, d_model
        self.mock_sae_model.encode = MagicMock(
            return_value=(None, torch.randn(1, 256))
        )  # _, h_post (batch, d_sae)

        # This self.mock_context will be used as the side_effect or return_value for the DASHBOARD_CONTEXT patch
        self.mock_context_object = SimpleNamespace()
        self.mock_context_object.primary_model = self.mock_primary_model
        self.mock_context_object.sae_model = self.mock_sae_model
        self.mock_context_object.pad_token_id = self.pad_token_id
        self.mock_context_object.device = self.device
        self.mock_context_object.vocab = {
            "AbilityX": 101,
            "AbilityY": 102,
            "AbilityKnown": 103,
        }
        self.mock_context_object.inv_vocab = {
            v: k for k, v in self.mock_context_object.vocab.items()
        }  # For display_primary_build tests

        # Setup feature_labels_manager mock on this object
        self.mock_context_object.feature_labels_manager = MagicMock()
        self.mock_context_object.feature_labels_manager.get_display_name = (
            MagicMock(side_effect=lambda x: f"Feature {x}")
        )

    # --- Tests for get_sae_activations ---

    def test_get_sae_activations_success(self):
        # Specific h_post for this test
        expected_h_post = torch.randn(1, 256)
        self.mock_sae_model.encode.return_value = (None, expected_h_post)

        ability_tokens = [101, 102, 103]
        weapon_token = 201

        result = get_sae_activations(
            self.mock_primary_model,
            self.mock_sae_model,
            ability_tokens,
            weapon_token,
            self.pad_token_id,
            self.device,
        )

        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, expected_h_post.squeeze()))
        self.mock_primary_model.token_embed.assert_called_once()
        self.mock_primary_model.weapon_embed.assert_called_once()
        self.mock_primary_model.input_proj.assert_called_once()
        self.mock_primary_model.transformer_layers[0].assert_called_once()
        self.mock_primary_model.masked_mean.assert_called_once()
        self.mock_sae_model.encode.assert_called_once()

    def test_get_sae_activations_no_primary_model(self):
        result = get_sae_activations(
            None,
            self.mock_sae_model,
            [101],
            201,
            self.pad_token_id,
            self.device,
        )
        self.assertIsNone(result)

    def test_get_sae_activations_no_sae_model(self):
        result = get_sae_activations(
            self.mock_primary_model,
            None,
            [101],
            201,
            self.pad_token_id,
            self.device,
        )
        self.assertIsNone(result)

    def test_get_sae_activations_model_internal_error(self):
        self.mock_primary_model.masked_mean.side_effect = Exception(
            "Internal model error"
        )
        result = get_sae_activations(
            self.mock_primary_model,
            self.mock_sae_model,
            [101],
            201,
            self.pad_token_id,
            self.device,
        )
        self.assertIsNone(result)

    # --- Tests for display_primary_build callback ---

    def test_display_primary_build_no_data(self):
        output_display, output_input_val = display_primary_build(None)
        self.assertEqual(output_display, "No primary build selected.")
        self.assertEqual(output_input_val, "")

    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_display_primary_build_with_data(self, mock_dashboard_context_obj):
        mock_dashboard_context_obj.inv_vocab = {
            101: "AbilityA",
            102: "AbilityB",
        }
        # Ensure other potentially accessed attributes are available or mocked if necessary
        # For display_primary_build, only inv_vocab is directly used.

        primary_data = {
            "weapon_id_token": "WeaponX",
            "ability_input_tokens": [101, 102],
            "activation": 0.75123,
        }

        output_display, output_input_val = display_primary_build(primary_data)

        self.assertIsInstance(output_display, list)
        self.assertIn("Weapon ID: WeaponX", str(output_display[0]))
        self.assertIn("Abilities: AbilityA, AbilityB", str(output_display[1]))
        self.assertIn("Original Activation: 0.7512", str(output_display[2]))
        self.assertEqual(output_input_val, "AbilityA, AbilityB")

    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_display_primary_build_unknown_token(
        self, mock_dashboard_context_obj
    ):
        mock_dashboard_context_obj.inv_vocab = {101: "AbilityA"}
        primary_data = {
            "weapon_id_token": "WeaponY",
            "ability_input_tokens": [101, 999],
            "activation": 0.5,
        }

        output_display, output_input_val = display_primary_build(primary_data)

        self.assertIn("Abilities: AbilityA, 999", str(output_display[1]))
        self.assertEqual(output_input_val, "AbilityA, 999")

    # --- Tests for run_ablation_analysis callback ---

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_no_models_in_context(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = None
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        result = run_ablation_analysis(
            n_clicks=1,
            primary_data={"weapon_id_token": 1},
            secondary_build_string="Test",
            selected_feature_id=0,
        )
        self.assertIsInstance(result, html.Div)
        self.assertIn("Models not loaded", result.children)

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_no_pad_id_in_context(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.pad_token_id = None
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        result = run_ablation_analysis(
            n_clicks=1,
            primary_data={"weapon_id_token": 1},
            secondary_build_string="Test",
            selected_feature_id=0,
        )
        self.assertIsInstance(result, html.Div)
        self.assertIn("PAD token ID not configured", result.children)

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_no_primary_data(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=None,
            secondary_build_string="AbilityX",
            selected_feature_id=0,
        )
        self.assertEqual(result, "Please select a primary build first.")

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_no_secondary_input(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [101]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string=" ",
            selected_feature_id=0,
        )
        self.assertEqual(result, "Please provide secondary ability tokens.")

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_get_primary_activations_fails(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        mock_get_sae_activations.return_value = None

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string="AbilityX",
            selected_feature_id=0,
        )
        self.assertEqual(
            result,
            "Could not compute primary activations. Check model and inputs.",
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_get_secondary_activations_fails(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        mock_get_sae_activations.side_effect = [torch.randn(256), None]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string="AbilityX,AbilityY",
            selected_feature_id=0,
        )

        self.assertIsInstance(result, html.Div)
        self.assertTrue(
            any(
                "Could not compute secondary activations" in str(child)
                for child in result.children
            )
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_success_feature_selected(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )
        mock_dashboard_context_obj.feature_labels_manager.get_display_name.return_value = (
            "Awesome Feature 1"
        )

        selected_idx = 1
        primary_acts = torch.tensor([0.1, 0.2, 0.9, 0.4])
        secondary_acts = torch.tensor([0.8, 0.15, 0.3, 0.41])
        mock_get_sae_activations.side_effect = [primary_acts, secondary_acts]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        secondary_input = "AbilityX,AbilityY"

        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string=secondary_input,
            selected_feature_id=selected_idx,
        )

        mock_get_sae_activations.assert_has_calls(
            [
                call(
                    self.mock_context_object.primary_model,
                    self.mock_context_object.sae_model,
                    [100],
                    1,
                    self.mock_context_object.pad_token_id,
                    self.mock_context_object.device,
                ),
                call(
                    self.mock_context_object.primary_model,
                    self.mock_context_object.sae_model,
                    [101, 102],
                    1,
                    self.mock_context_object.pad_token_id,
                    self.mock_context_object.device,
                ),
            ]
        )
        mock_dashboard_context_obj.feature_labels_manager.get_display_name.assert_called_once_with(
            selected_idx
        )

        self.assertIsInstance(result, html.Div)
        result_str = str(result)
        self.assertIn("Ablation for Awesome Feature 1:", result_str)
        self.assertIn(
            f"Primary Activation: {primary_acts[selected_idx]:.4f}", result_str
        )
        self.assertIn(
            f"Secondary Activation: {secondary_acts[selected_idx]:.4f}",
            result_str,
        )
        self.assertIn(
            f"Difference: {secondary_acts[selected_idx] - primary_acts[selected_idx]:.4f}",
            result_str,
        )
        self.assertNotIn("Top Feature Activation Changes", result_str)

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_no_feature_selected(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        primary_acts = torch.randn(4)
        secondary_acts = torch.randn(4)
        mock_get_sae_activations.side_effect = [primary_acts, secondary_acts]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string="AbilityX",
            selected_feature_id=None,
        )

        self.assertIsInstance(result, html.Div)
        self.assertIn(
            "Please select a feature from the dropdown",
            str(result.children[0].children),
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_selected_feature_out_of_bounds(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        primary_acts = torch.randn(4)
        secondary_acts = torch.randn(4)
        mock_get_sae_activations.side_effect = [primary_acts, secondary_acts]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string="AbilityX",
            selected_feature_id=10,
        )

        self.assertIsInstance(result, html.Div)
        self.assertIn(
            "Error: Selected feature ID 10 is out of bounds", result.children
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_selected_feature_invalid_format(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        primary_acts = torch.randn(4)
        secondary_acts = torch.randn(4)
        mock_get_sae_activations.side_effect = [primary_acts, secondary_acts]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string="AbilityX",
            selected_feature_id="not-an-int",
        )

        self.assertIsInstance(result, html.Div)
        self.assertIn(
            "Error: Invalid feature ID format: not-an-int", result.children
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_feature_labels_manager_missing(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = self.mock_context_object.vocab
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = None

        selected_idx = 0
        primary_acts = torch.tensor([0.1, 0.2])
        secondary_acts = torch.tensor([0.3, 0.4])
        mock_get_sae_activations.side_effect = [primary_acts, secondary_acts]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string="AbilityX",
            selected_feature_id=selected_idx,
        )

        self.assertIsInstance(result, html.Div)
        result_str = str(result)
        self.assertIn(f"Ablation for Feature {selected_idx}:", result_str)
        self.assertIn(
            f"Primary Activation: {primary_acts[selected_idx]:.4f}", result_str
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_unknown_secondary_tokens(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = {"AbilityKnown": 103}
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )
        mock_dashboard_context_obj.feature_labels_manager.get_display_name.return_value = (
            "Known Feature 0"
        )

        selected_idx = 0
        primary_acts = torch.tensor([0.1, 0.2])
        secondary_acts = torch.tensor([0.3, 0.4])
        mock_get_sae_activations.side_effect = [primary_acts, secondary_acts]

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        secondary_input = "AbilityKnown,UnknownAbility"

        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string=secondary_input,
            selected_feature_id=selected_idx,
        )

        self.assertIsInstance(result, html.Div)
        self.assertIn(
            "Warning: The following ability names were not found in vocabulary and will be skipped: UnknownAbility",
            str(result.children[0]),
        )
        focused_result_div = result.children[1]
        self.assertIn(
            "Ablation for Known Feature 0:", str(focused_result_div.children[0])
        )

        mock_get_sae_activations.assert_has_calls(
            [
                call(
                    self.mock_context_object.primary_model,
                    self.mock_context_object.sae_model,
                    [100],
                    1,
                    self.mock_context_object.pad_token_id,
                    self.mock_context_object.device,
                ),
                call(
                    self.mock_context_object.primary_model,
                    self.mock_context_object.sae_model,
                    [103],
                    1,
                    self.mock_context_object.pad_token_id,
                    self.mock_context_object.device,
                ),
            ]
        )

    @patch(
        "splatnlp.dashboard.components.ablation_component.get_sae_activations"
    )
    @patch("splatnlp.dashboard.app.DASHBOARD_CONTEXT")
    def test_run_ablation_no_valid_secondary_tokens_after_parse(
        self, mock_dashboard_context_obj, mock_get_sae_activations
    ):
        mock_dashboard_context_obj.primary_model = (
            self.mock_context_object.primary_model
        )
        mock_dashboard_context_obj.sae_model = (
            self.mock_context_object.sae_model
        )
        mock_dashboard_context_obj.vocab = {}
        mock_dashboard_context_obj.pad_token_id = (
            self.mock_context_object.pad_token_id
        )
        mock_dashboard_context_obj.device = self.mock_context_object.device
        mock_dashboard_context_obj.feature_labels_manager = (
            self.mock_context_object.feature_labels_manager
        )

        primary_acts = torch.tensor([0.1, 0.2])
        mock_get_sae_activations.return_value = primary_acts

        primary_data = {"weapon_id_token": 1, "ability_input_tokens": [100]}
        secondary_input = "Unknown1,Unknown2"

        result = run_ablation_analysis(
            n_clicks=1,
            primary_data=primary_data,
            secondary_build_string=secondary_input,
            selected_feature_id=0,
        )

        self.assertIsInstance(result, html.Div)
        self.assertTrue(
            any(
                "Warning: The following ability names were not found"
                in str(child)
                for child in result.children
                if isinstance(child, html.P)
            )
        )
        self.assertTrue(
            any(
                "No valid secondary ability tokens to process" in str(child)
                for child in result.children
                if isinstance(child, html.P)
            )
        )

        mock_get_sae_activations.assert_called_once_with(
            self.mock_context_object.primary_model,
            self.mock_context_object.sae_model,
            [100],
            1,
            self.mock_context_object.pad_token_id,
            self.mock_context_object.device,
        )


if __name__ == "__main__":
    unittest.main()
