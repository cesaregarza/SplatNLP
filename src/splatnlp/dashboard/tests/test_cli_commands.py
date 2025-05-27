import argparse
import json
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import joblib

# Assuming the commands and database_manager are importable
# This might require adjusting PYTHONPATH or the test execution environment
from splatnlp.dashboard.commands.consolidate_to_db_cmd import consolidate_to_db_command
from splatnlp.dashboard.commands.generate_activations_cmd import generate_activations_command
# Import other command functions as needed for invocation tests
# from splatnlp.dashboard.commands.extract_top_examples_cmd import extract_top_examples_command
# from splatnlp.dashboard.commands.compute_correlations_efficient_cmd import compute_correlations_efficient_command
# ... and so on for other commands

from splatnlp.dashboard.database_manager import DashboardDatabase
from splatnlp.dashboard.cli import main as cli_main # To test parser and command dispatch

class TestConsolidateToDbCommand(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to act as the root for mock files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Mock database path
        self.mock_db_path = self.temp_path / "test_dashboard.sqlite"

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    @mock.patch('splatnlp.dashboard.commands.consolidate_to_db_cmd.DashboardDatabase')
    def test_consolidate_all_inputs_provided(self, MockDashboardDatabase):
        # Mock the database instance and its methods
        mock_db_instance = MockDashboardDatabase.return_value
        
        # Create mock input files with sample data
        correlations_data = {
            "correlations": [
                {"neuron_i": 0, "neuron_j": 1, "correlation": 0.5, "n_common": 10},
                {"neuron_i": 1, "neuron_j": 2, "correlation": -0.3, "n_common": 5}
            ]
        }
        mock_correlations_path = self.temp_path / "correlations.json"
        with open(mock_correlations_path, 'w') as f:
            json.dump(correlations_data, f)

        analytics_data = {
            "features": [
                {
                    "id": 0,
                    "statistics": {"mean": 0.1, "std": 0.05, "min": 0.0, "max": 0.5, "median": 0.08, "q25": 0.02, "q75": 0.15, "n_zeros": 5, "n_total": 100, "sparsity": 0.05, "histogram": {"counts": [10,20], "bin_edges": [0,0.1,0.2]}},
                    "top_logit_influences": {"positive": [{"token_name": "A", "influence": 0.9}], "negative": [{"token_name": "B", "influence": -0.8}]},
                    "top_activating_examples": [{"original_index": 101, "rank": 1, "activation_value": 0.95}]
                }
            ]
        }
        mock_analytics_path = self.temp_path / "analytics.joblib"
        joblib.dump(analytics_data, mock_analytics_path)

        args = argparse.Namespace(
            database_path=str(self.mock_db_path),
            input_correlations_path=str(mock_correlations_path),
            input_precomputed_analytics_path=str(mock_analytics_path)
        )

        consolidate_to_db_command(args)

        # Assert DashboardDatabase was instantiated
        MockDashboardDatabase.assert_called_once_with(self.mock_db_path)

        # Assert that data insertion methods were called on the instance
        # Check calls to executemany with expected data structures
        # This requires inspecting mock_db_instance.get_connection.return_value.executemany.call_args_list

        # Example assertion for correlations (structure might need adjustment based on actual implementation)
        # We expect two calls for INSERT OR REPLACE INTO feature_correlations
        # This is a simplified check; more rigorous checks would inspect the actual data passed.
        found_correlations_insert = False
        found_stats_insert = False
        found_logits_insert = False
        found_top_examples_insert = False

        for call in mock_db_instance.get_connection.return_value.executemany.call_args_list:
            sql_query = call.args[0]
            if "INSERT OR REPLACE INTO feature_correlations" in sql_query:
                found_correlations_insert = True
                self.assertEqual(len(call.args[1]), 2) # Two correlation items
            elif "INSERT OR REPLACE INTO feature_stats" in sql_query:
                found_stats_insert = True
                self.assertEqual(len(call.args[1]), 1) # One feature's stats
            elif "INSERT OR REPLACE INTO logit_influences" in sql_query:
                found_logits_insert = True
                self.assertEqual(len(call.args[1]), 2) # Two logit influence items (1 pos, 1 neg)
            elif "INSERT OR REPLACE INTO top_examples" in sql_query:
                found_top_examples_insert = True
                self.assertEqual(len(call.args[1]), 1) # One top example
        
        self.assertTrue(found_correlations_insert, "Feature correlations insert not called.")
        self.assertTrue(found_stats_insert, "Feature stats insert not called.")
        self.assertTrue(found_logits_insert, "Logit influences insert not called.")
        self.assertTrue(found_top_examples_insert, "Top examples insert not called.")
        
        mock_db_instance.vacuum.assert_called_once()


    @mock.patch('splatnlp.dashboard.commands.consolidate_to_db_cmd.DashboardDatabase')
    def test_consolidate_missing_analytics_file(self, MockDashboardDatabase):
        mock_db_instance = MockDashboardDatabase.return_value
        
        mock_correlations_path = self.temp_path / "correlations.json"
        # Analytics file is deliberately not created
        # with open(mock_correlations_path, 'w') as f: # Create dummy correlations
        #     json.dump({"correlations": []}, f)


        args = argparse.Namespace(
            database_path=str(self.mock_db_path),
            input_correlations_path=None, #str(mock_correlations_path),
            input_precomputed_analytics_path=str(self.temp_path / "non_existent_analytics.joblib")
        )

        consolidate_to_db_command(args)
        
        MockDashboardDatabase.assert_called_once_with(self.mock_db_path)
        # Check that inserts for analytics were not called or called with empty data
        # For example, feature_stats insert should not happen if analytics file is missing
        for call in mock_db_instance.get_connection.return_value.executemany.call_args_list:
            sql_query = call.args[0]
            self.assertNotIn("INSERT OR REPLACE INTO feature_stats", sql_query)
            self.assertNotIn("INSERT OR REPLACE INTO logit_influences", sql_query)
            self.assertNotIn("INSERT OR REPLACE INTO top_examples", sql_query)
        
        mock_db_instance.vacuum.assert_called_once() # Vacuum should still be called

    @mock.patch('splatnlp.dashboard.commands.consolidate_to_db_cmd.DashboardDatabase')
    def test_consolidate_no_input_files_provided(self, MockDashboardDatabase):
        mock_db_instance = MockDashboardDatabase.return_value
        args = argparse.Namespace(
            database_path=str(self.mock_db_path),
            input_correlations_path=None,
            input_precomputed_analytics_path=None
        )
        consolidate_to_db_command(args)
        MockDashboardDatabase.assert_called_once_with(self.mock_db_path)
        
        # Assert that no data insertion methods were called if no input files
        mock_db_instance.get_connection.return_value.executemany.assert_not_called()
        mock_db_instance.vacuum.assert_called_once()


class TestGenerateActivationsCommandInvocation(unittest.TestCase):

    @mock.patch('splatnlp.dashboard.commands.generate_activations_cmd.generate_activations_command')
    @mock.patch('argparse.ArgumentParser.parse_args')
    def test_cli_invokes_generate_activations(self, mock_parse_args, mock_generate_activations_func):
        # Simulate CLI arguments for 'generate-activations'
        # These need to be strings, as if from command line
        cli_args = [
            'generate-activations',
            '--primary-model-checkpoint', 'mock_primary.pth',
            '--sae-model-checkpoint', 'mock_sae.pth',
            '--vocab-path', 'mock_vocab.json',
            '--weapon-vocab-path', 'mock_weapon_vocab.json',
            '--data-path', 'mock_data.csv',
            '--output-path', 'mock_output_cache.joblib',
            # Add other required args with dummy values
            '--embedding-dim', '32',
            '--hidden-dim', '512',
            '--num-layers', '3',
            '--num-heads', '8',
            '--num-inducing-points', '16', # Example value
            '--sae-expansion-factor', '4', # Example value
        ]
        
        # The cli.py's main function creates its own parser.
        # We mock parse_args for that parser.
        # The key is that the `func` attribute of the parsed args
        # should point to the command function we are testing.
        
        # To properly test the parser setup in cli.py, we would let it parse,
        # then check the 'func' attribute. Here, we simplify by assuming the
        # parser correctly maps subcommand to func.
        
        # Let's make mock_parse_args return an object that has 'func' set to our target command
        # and other necessary attributes.
        
        # Create a mock namespace that `cli_main` would produce after parsing.
        # The `func` attribute is key.
        mock_args_obj = argparse.Namespace(
            main_command='generate-activations', # As set by dest='main_command'
            func=generate_activations_command, # This is what cli.py's parser should set
            primary_model_checkpoint='mock_primary.pth',
            sae_model_checkpoint='mock_sae.pth',
            vocab_path='mock_vocab.json',
            weapon_vocab_path='mock_weapon_vocab.json',
            data_path='mock_data.csv',
            output_path='mock_output_cache.joblib',
            fraction=0.1, chunk_size=0.01, chunk_storage_dir="activation_chunks_tmp", random_state=42,
            embedding_dim=32, hidden_dim=512, num_layers=3, num_heads=8,
            num_inducing_points=16, sae_expansion_factor=4.0, batch_size=64,
            hook_target="masked_mean", force=False
        )
        mock_parse_args.return_value = mock_args_obj

        # Call the main CLI entry point with mocked sys.argv
        with mock.patch('sys.argv', ['cli.py'] + cli_args):
            cli_main()

        # Assert that our specific command function was called with the parsed args
        mock_generate_activations_func.assert_called_once_with(mock_args_obj)


if __name__ == '__main__':
    unittest.main()
