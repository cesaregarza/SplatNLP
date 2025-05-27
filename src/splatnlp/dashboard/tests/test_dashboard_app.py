import unittest
import json
from unittest import mock
from pathlib import Path
import tempfile
import shutil # For cleaning up temp directory if needed, though tempfile.TemporaryDirectory handles it

import dash
from dash import html, dcc
import numpy as np
import pandas as pd # For creating Series for _example_card if used directly

# Application imports
from splatnlp.dashboard.app import app as actual_app_instance # The Dash app
from splatnlp.dashboard.app import DASHBOARD_CONTEXT as app_context_ref # Global context
from splatnlp.dashboard.database_manager import DashboardDatabase, DatabaseBackedContext

# Import component callback functions or data processing functions to be tested
# These are the functions that directly interact with app_context_ref.db_context
from splatnlp.dashboard.components.activation_hist import update_activation_histogram
from splatnlp.dashboard.components.correlations_component import update_correlations_display
from splatnlp.dashboard.components.top_examples_component import update_top_examples_grid
from splatnlp.dashboard.components.intervals_grid_component import render_intervals_grid, _example_card
from splatnlp.dashboard.components.feature_labels import FeatureLabelsManager # For context
from splatnlp.preprocessing.transform.mappings import generate_maps # For id_to_name in intervals_grid

# Mock data for vocabularies
MOCK_VOCAB = {"<PAD>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5, "dog": 6, "ate": 7, "food": 8, "A":9, "B":10}
MOCK_INV_VOCAB = {str(v): k for k, v in MOCK_VOCAB.items()} # Ensure string keys for inv_vocab
MOCK_WEAPON_VOCAB = {"Splatana Wiper": 0, "Splattershot": 1}
MOCK_INV_WEAPON_VOCAB = {v: k for k, v in MOCK_WEAPON_VOCAB.items()}


class TestDashboardAppDataSourceDB(unittest.TestCase):
    """
    Tests for dashboard components focusing on data retrieval via DatabaseBackedContext.
    """

    def setUp(self):
        """
        Set up a temporary database with sample data and configure the app context.
        """
        # Create a temporary directory to hold the database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_dashboard.sqlite"

        # Initialize DashboardDatabase (this creates the schema)
        self.db_manager = DashboardDatabase(self.db_path)

        # Populate the database with sample data
        self._populate_sample_data()

        # Create DatabaseBackedContext
        self.db_context = DatabaseBackedContext(self.db_manager)

        # Set up the global app_context_ref to use the db_context
        # This mocks the state of DASHBOARD_CONTEXT after cli.py's load_dashboard_data
        app_context_ref.db_context = self.db_context
        app_context_ref.vocab = MOCK_VOCAB
        app_context_ref.inv_vocab = MOCK_INV_VOCAB
        app_context_ref.weapon_vocab = MOCK_WEAPON_VOCAB
        app_context_ref.inv_weapon_vocab = MOCK_INV_WEAPON_VOCAB
        app_context_ref.feature_labels_manager = FeatureLabelsManager() # Init empty
        app_context_ref.primary_model = None # Not testing dynamic tooltips here
        app_context_ref.sae_model = None     # Not testing dynamic tooltips here
        app_context_ref.precomputed_analytics = None # Explicitly None
        app_context_ref.device = "cpu"
        
        # For intervals_grid_component, generate_maps() is called.
        # It might have its own dependencies or be mocked if it's too complex.
        # For now, assume it works or mock its output if needed.
        _, self.id_to_name_map, _ = generate_maps()


    def _populate_sample_data(self):
        """
        Helper method to insert sample data into the database tables.
        """
        with self.db_manager.get_connection() as conn:
            # Sample Examples
            sample_examples_data = [
                (0, 0, "Splatana Wiper", json.dumps([1,2,3]), "the cat sat", "the cat sat on mat", False, json.dumps({"source": "test"})),
                (1, 1, "Splattershot", json.dumps([6,7,8]), "dog ate food", "dog ate cat food", False, json.dumps({"source": "test"})),
                (2, 0, "Splatana Wiper", json.dumps([4,5]), "on mat", "on the mat", False, json.dumps({"source": "test"})),
                (3, 1, "Splattershot", json.dumps([1,8]), "the food", "the good food", False, json.dumps({"source": "test"})),
            ]
            conn.executemany("INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_examples_data)

            # Sample Activations (sparse)
            # (example_id, feature_id, activation_value)
            sample_activations_data = [
                (0, 0, 0.95), (0, 1, 0.1),
                (1, 0, 0.05), (1, 1, 0.88), (1, 2, 0.5),
                (2, 0, 0.60), (2, 2, 0.75),
                (3, 1, 0.30), (3, 2, 0.02),
            ]
            conn.executemany("INSERT INTO activations VALUES (?, ?, ?)", sample_activations_data)

            # Sample Feature Statistics (for feature 0, 1, 2)
            # histogram_data is JSON string: {'counts': [], 'bin_edges': []}
            hist0 = json.dumps({'counts': [1,0,1,1], 'bin_edges': [0.0, 0.25, 0.5, 0.75, 1.0]})
            hist1 = json.dumps({'counts': [1,1,0,1], 'bin_edges': [0.0, 0.25, 0.5, 0.75, 1.0]})
            hist2 = json.dumps({'counts': [1,0,1], 'bin_edges': [0.0, 0.3, 0.6, 0.9]})
            sample_feature_stats_data = [
                (0, 0.53, 0.1, 0.05, 0.95, 0.5, 0.1, 0.7, 0, 3, 0.0, hist0),
                (1, 0.43, 0.2, 0.1, 0.88, 0.4, 0.2, 0.6, 0, 3, 0.0, hist1),
                (2, 0.62, 0.15, 0.02, 0.75, 0.6, 0.3, 0.7, 1, 3, 0.33, hist2),
            ]
            conn.executemany("INSERT INTO feature_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_feature_stats_data)

            # Sample Top Examples (for feature 0, 1, 2)
            # (feature_id, example_id, rank, activation_value)
            sample_top_examples_data = [
                (0, 0, 1, 0.95), (0, 2, 2, 0.60), (0, 1, 3, 0.05), # Feature 0
                (1, 1, 1, 0.88), (1, 3, 2, 0.30), (1, 0, 3, 0.10), # Feature 1
                (2, 2, 1, 0.75), (2, 1, 2, 0.50), (2, 3, 3, 0.02), # Feature 2
            ]
            conn.executemany("INSERT INTO top_examples VALUES (?, ?, ?, ?)", sample_top_examples_data)

            # Sample Feature Correlations (symmetric, feature_a < feature_b)
            # (feature_a, feature_b, correlation)
            sample_feature_correlations_data = [
                (0, 1, 0.2),
                (0, 2, -0.5),
                (1, 2, 0.7),
            ]
            conn.executemany("INSERT INTO feature_correlations VALUES (?, ?, ?)", sample_feature_correlations_data)

            # Sample Logit Influences (for feature 0, 1)
            # (feature_id, token_id, token_name, influence, rank)
            sample_logit_influences_data = [
                (0, MOCK_VOCAB["cat"], "cat", 0.9, 1), (0, MOCK_VOCAB["mat"], "mat", 0.7, 2), # Feature 0 Positive
                (0, MOCK_VOCAB["dog"], "dog", -0.8, 3), # Feature 0 Negative (rank continues)
                (1, MOCK_VOCAB["food"], "food", 0.85, 1), (1, MOCK_VOCAB["ate"], "ate", 0.75, 2), # Feature 1 Positive
                (1, MOCK_VOCAB["sat"], "sat", -0.6, 3), # Feature 1 Negative
            ]
            conn.executemany("INSERT INTO logit_influences VALUES (?, ?, ?, ?, ?)", sample_logit_influences_data)
            
            conn.commit()

    def tearDown(self):
        """
        Clean up temporary directory and files.
        """
        # Clear the context references
        app_context_ref.db_context = None
        app_context_ref.vocab = None
        app_context_ref.inv_vocab = None
        # ... any other context attributes that were set ...
        self.temp_dir.cleanup()

    def test_activation_histogram_data_retrieval(self):
        """Test that activation_hist.py's callback fetches data correctly."""
        # Test for feature 0
        # The callback update_activation_histogram(selected_feature_id, filter_type, _)
        # The third argument `_` is State("feature-dropdown", "value"), which is the same as selected_feature_id
        figure_data_all = update_activation_histogram(0, "all", 0)
        self.assertIsNotNone(figure_data_all)
        self.assertTrue(len(figure_data_all['data']) > 0, "Histogram data should not be empty for 'all'")
        self.assertEqual(figure_data_all['layout']['title'], "All Activations for Feature 0")

        # Check some data points if possible - depends on exact histogram binning
        # For example, check if counts match what was inserted in feature_stats
        expected_hist0 = json.loads(json.dumps({'counts': [1,0,1,1], 'bin_edges': [0.0, 0.25, 0.5, 0.75, 1.0]})) # from setup
        self.assertEqual(list(figure_data_all['data'][0]['y']), expected_hist0['counts'])

        figure_data_nonzero = update_activation_histogram(0, "nonzero", 0)
        self.assertIsNotNone(figure_data_nonzero)
        # Non-zero test would depend on actual values vs 1e-6 and binning.
        # For feature 0: values are 0.95, 0.05, 0.60. All > 1e-6.
        # The exact output for 'nonzero' depends on how bin_edges are filtered.
        # For now, just check that it runs and produces some output.
        self.assertTrue(len(figure_data_nonzero['data']) > 0, "Histogram data should not be empty for 'nonzero'")
        self.assertEqual(figure_data_nonzero['layout']['title'], "Non-Zero Activations for Feature 0")


    def test_correlations_component_data_retrieval(self):
        """Test that correlations_component.py's callback fetches data correctly."""
        # Test for feature 0
        # update_correlations_display(selected_feature_id)
        sae_corr_display, token_logit_display, error_msg = update_correlations_display(0)
        
        self.assertEqual(error_msg, "")
        self.assertTrue(len(sae_corr_display) > 1, "SAE feature correlations display should not be empty.")
        # Expected for feature 0: corr with 1 (0.2), corr with 2 (-0.5)
        # The display is html.P elements. Check their content.
        self.assertIn("Feature 1: 0.200", str(sae_corr_display[1])) # Assuming H5 title is first
        self.assertIn("Feature 2: -0.500", str(sae_corr_display[2]))

        self.assertTrue(len(token_logit_display) > 1, "Token logit influences display should not be empty.")
        # Expected for feature 0: cat (0.9), mat (0.7), dog (-0.8)
        self.assertIn("Token 'cat': Influence 0.900", str(token_logit_display[1]))
        self.assertIn("Token 'mat': Influence 0.700", str(token_logit_display[2]))
        self.assertIn("Token 'dog': Influence -0.800", str(token_logit_display[3]))

    def test_top_examples_component_data_retrieval(self):
        """Test that top_examples_component.py's callback fetches data correctly."""
        # Test for feature 1
        # update_top_examples_grid(selected_feature_id)
        grid_data, _, error_msg = update_top_examples_grid(1)
        self.assertEqual(error_msg, "")
        self.assertEqual(len(grid_data), 3) # Expecting 3 top examples for feature 1
        
        # Check data for the first top example for feature 1 (example_id 1, rank 1, act 0.88)
        self.assertEqual(grid_data[0]['Rank'], 1)
        self.assertEqual(grid_data[0]['Original Index'], 1) # example_id
        self.assertEqual(grid_data[0]['SAE Feature Activation'], "0.8800")
        self.assertEqual(grid_data[0]['Weapon'], "Splattershot") # From example_id 1
        self.assertEqual(grid_data[0]['Input Abilities'], "dog ate food") # From example_id 1

    def test_intervals_grid_component_data_retrieval(self):
        """Test intervals_grid_component.py's callback data fetching (simplified)."""
        # Test for feature 2
        # render_intervals_grid(selected_feature_id)
        # This component is more complex due to random sampling and card generation.
        # We'll do a basic check: ensure it runs, and some cards are generated.
        # The TF-IDF part is skipped when using db_context in the refactored component.
        
        sections, error_msg = render_intervals_grid(2)
        self.assertEqual(error_msg, "")
        self.assertTrue(len(sections) > 0, "Intervals grid sections should not be empty.")
        
        # Check for at least one interval header and some cards
        found_interval_header = False
        found_card = False
        for section_div in sections:
            if isinstance(section_div, html.Div):
                for child in section_div.children:
                    if isinstance(child, html.Div) and "Interval" in str(child.children):
                        found_interval_header = True
                    if isinstance(child, dash.html.Div) and hasattr(child, 'children'): # Looking for dbc.Row containing cards
                         if child.children and isinstance(child.children[0], dash.html.Div): # dbc.Col
                              if hasattr(child.children[0],'children') and "Card" in str(type(child.children[0].children)):
                                   found_card = True
                                   break
                if found_card: break
        
        self.assertTrue(found_interval_header, "Did not find any interval headers.")
        self.assertTrue(found_card, "Did not find any example cards in intervals.")

        # Test _example_card directly as it's a critical part of intervals_grid
        sample_example_data_from_db = {
            'id': 0, 'weapon_id_token': 0, 'weapon_name': "Splatana Wiper",
            'ability_input_tokens': [1,2,3], # "the cat sat"
            'input_abilities_str': "the cat sat",
            'top_predicted_abilities_str': "the cat sat on mat",
            'is_null_token': False, 'metadata': "{}"
        }
        record_series = pd.Series(sample_example_data_from_db)
        card = _example_card(record_series, MOCK_INV_VOCAB, MOCK_INV_WEAPON_VOCAB, 0.95, self.id_to_name_map, set())
        self.assertIsNotNone(card)
        self.assertIn("Splatana Wiper", str(card))
        self.assertIn("the cat sat", str(card))
        self.assertIn("0.9500", str(card))


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # For Dash testing, app context needs to be set up before callbacks are registered.
    # However, we are testing the callbacks as functions here, providing context directly.
    # If we were testing via app.test_client(), app initialization order would be more critical.
    unittest.main()
