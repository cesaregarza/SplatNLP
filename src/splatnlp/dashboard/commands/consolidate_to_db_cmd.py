"""
Command for consolidating pre-computed analytics data into the SQLite database.
This command reads data from various files (often outputs of other CLI commands)
and populates the relevant tables in the dashboard database.
"""

import argparse
import json
import logging
import pickle  # For .pkl files
from pathlib import Path
from typing import Optional

import joblib  # For .joblib files which are common for sklearn/numpy related data
import numpy as np
import pandas as pd  # If some inputs are tabular and easily read by pandas

# Assuming DashboardDatabase is accessible.
try:
    from splatnlp.dashboard.database_manager import DashboardDatabase
except ImportError:
    # Fallback for environments where the package isn't fully set up,
    # allowing the script to be parsed but likely fail at runtime if DB is used.
    logging.getLogger(__name__).critical(
        "Failed to import DashboardDatabase. Ensure splatnlp package is correctly installed."
    )

    # Define a dummy class to prevent import errors during parsing in such environments
    class DashboardDatabase:
        def __init__(self, db_path):
            pass

        def get_connection(self):
            raise RuntimeError("Dummy DashboardDatabase")

        # Add other methods that might be called to prevent AttributeError during parsing


logger = logging.getLogger(__name__)


def load_json_data(
    path: Optional[Path], expected_type=dict, description: str = "data"
):
    """Loads JSON data from a file if path is provided."""
    if not path:
        logger.info(f"{description} path not provided, skipping.")
        return None
    if not path.exists():
        logger.warning(f"{description} file not found at {path}, skipping.")
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, expected_type):
            logger.warning(
                f"{description} at {path} is not of expected type {expected_type.__name__}. Got {type(data).__name__}."
            )
            return None
        logger.info(f"Successfully loaded {description} from {path}")
        return data
    except Exception as e:
        logger.error(
            f"Error loading {description} from {path}: {e}", exc_info=True
        )
        return None


def load_joblib_data(path: Optional[Path], description: str = "data"):
    """Loads joblib data from a file if path is provided."""
    if not path:
        logger.info(f"{description} path not provided, skipping.")
        return None
    if not path.exists():
        logger.warning(f"{description} file not found at {path}, skipping.")
        return None
    try:
        data = joblib.load(path)
        logger.info(f"Successfully loaded {description} from {path}")
        return data
    except Exception as e:
        logger.error(
            f"Error loading {description} from {path}: {e}", exc_info=True
        )
        return None


def consolidate_to_db_command(args: argparse.Namespace):
    """
    Consolidates various pre-computed data files into the SQLite database.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info(
        f"Starting consolidation of analytics data into database: {args.database_path}"
    )

    db_path = Path(args.database_path)
    db = DashboardDatabase(db_path)

    # 1. Consolidate Feature Correlations
    # Expected format: JSON file with a list of dicts, each like:
    # {'neuron_i': int, 'neuron_j': int, 'correlation': float, 'n_common': int}
    # Or the format saved by compute_correlations_efficient_cmd:
    # {'correlations': [{'neuron_i': ..., 'neuron_j': ..., 'correlation': ...}]}
    correlations_data_wrapper = load_json_data(
        (
            Path(args.input_correlations_path)
            if args.input_correlations_path
            else None
        ),
        description="correlations data",
    )
    if (
        correlations_data_wrapper
        and "correlations" in correlations_data_wrapper
    ):
        correlations_list = correlations_data_wrapper["correlations"]
        if isinstance(correlations_list, list):
            db_corr_data = []
            for corr_item in correlations_list:
                if all(
                    k in corr_item
                    for k in ["neuron_i", "neuron_j", "correlation"]
                ):
                    # Store feature_a < feature_b to avoid duplicates if not already handled
                    feat_a = min(corr_item["neuron_i"], corr_item["neuron_j"])
                    feat_b = max(corr_item["neuron_i"], corr_item["neuron_j"])
                    db_corr_data.append(
                        (feat_a, feat_b, corr_item["correlation"])
                    )
                else:
                    logger.warning(
                        f"Skipping malformed correlation item: {corr_item}"
                    )

            if db_corr_data:
                try:
                    with db.get_connection() as conn:
                        # Consider deleting old correlations for these pairs or all? For now, INSERT OR REPLACE.
                        conn.executemany(
                            "INSERT OR REPLACE INTO feature_correlations (feature_a, feature_b, correlation) VALUES (?, ?, ?)",
                            db_corr_data,
                        )
                        conn.commit()
                    logger.info(
                        f"Consolidated {len(db_corr_data)} items into 'feature_correlations' table."
                    )
                except Exception as e:
                    logger.error(
                        f"Error inserting feature correlations: {e}",
                        exc_info=True,
                    )
        else:
            logger.warning(
                "Correlations data is not a list as expected in the JSON structure."
            )

    # 2. Consolidate Feature Statistics & Logit Influences & Top Examples (from precompute_analytics output)
    # Expected format: A .joblib file (dict) with a top-level key 'features',
    # which is a list of dicts, one per feature/neuron.
    # Each feature dict contains 'id', 'statistics', 'top_logit_influences', 'top_activating_examples'.
    # Note: The 'extract-top-examples' command saves per-neuron JSONs. This command
    # assumes a single consolidated analytics file from 'precompute-analytics'.
    # If individual neuron files are the source, this part needs to glob and iterate.
    # For now, assuming the consolidated file from 'precompute_analytics_cmd'.

    analytics_data_path = (
        Path(args.input_precomputed_analytics_path)
        if args.input_precomputed_analytics_path
        else None
    )
    all_features_analytics = load_joblib_data(
        analytics_data_path, description="precomputed analytics data"
    )

    if (
        all_features_analytics
        and "features" in all_features_analytics
        and isinstance(all_features_analytics["features"], list)
    ):
        feature_stats_batch = []
        logit_influences_batch = []
        top_examples_batch = []

        for feature_data in all_features_analytics["features"]:
            feature_id = feature_data.get("id")
            if feature_id is None:
                logger.warning(
                    f"Skipping feature data with missing 'id': {feature_data}"
                )
                continue

            # Consolidate Feature Statistics
            stats = feature_data.get("statistics")
            if stats and isinstance(stats, dict):
                hist_data_json = json.dumps(
                    stats.get("histogram", {})
                )  # Ensure histogram is JSON string
                feature_stats_batch.append(
                    (
                        feature_id,
                        stats.get("mean"),
                        stats.get("std"),
                        stats.get(
                            "min_val", stats.get("min")
                        ),  # precompute_analytics uses 'min'
                        stats.get(
                            "max_val", stats.get("max")
                        ),  # precompute_analytics uses 'max'
                        stats.get("median"),
                        stats.get("q25"),
                        stats.get("q75"),
                        stats.get("n_zeros"),
                        stats.get("n_total"),
                        stats.get("sparsity"),
                        hist_data_json,
                    )
                )

            # Consolidate Logit Influences
            logit_influences = feature_data.get("top_logit_influences")
            if logit_influences and isinstance(logit_influences, dict):
                # Positive influences
                for rank_idx, influence_item in enumerate(
                    logit_influences.get("positive", []), 1
                ):
                    # Assuming token_name is available, token_id might not be directly.
                    # The DB schema has token_id and token_name. If only name, might need vocab lookup.
                    # For now, store token_name, and use rank. Token_id can be null if not available.
                    logit_influences_batch.append(
                        (
                            feature_id,
                            None,  # token_id - might need to be derived or schema adapted
                            influence_item.get("token_name"),
                            influence_item.get("influence"),
                            rank_idx,  # Rank for positive
                        )
                    )
                # Negative influences - store with adjusted rank or a type indicator
                for rank_idx, influence_item in enumerate(
                    logit_influences.get("negative", []), 1
                ):
                    logit_influences_batch.append(
                        (
                            feature_id,
                            None,
                            influence_item.get("token_name"),
                            influence_item.get(
                                "influence"
                            ),  # Should be negative already if stored directly
                            rank_idx
                            + len(
                                logit_influences.get("positive", [])
                            ),  # Continue rank for negatives
                        )
                    )

            # Consolidate Top Activating Examples
            # This data comes from precompute_analytics, which has a slightly different format
            # than the output of extract_top_examples (which saves per-neuron JSONs).
            # The DB schema for top_examples is (feature_id, example_id, rank, activation_value)
            top_examples_data = feature_data.get("top_activating_examples")
            if top_examples_data and isinstance(top_examples_data, list):
                for ex_item in top_examples_data:
                    if all(
                        k in ex_item
                        for k in ["original_index", "rank", "activation_value"]
                    ):
                        top_examples_batch.append(
                            (
                                feature_id,
                                ex_item[
                                    "original_index"
                                ],  # This is the example_id
                                ex_item["rank"],
                                ex_item["activation_value"],
                            )
                        )
                    else:
                        logger.warning(
                            f"Skipping malformed top_example item for feature {feature_id}: {ex_item}"
                        )

        # Batch insert into DB
        try:
            with db.get_connection() as conn:
                if feature_stats_batch:
                    conn.executemany(
                        """INSERT OR REPLACE INTO feature_stats
                           (feature_id, mean, std, min_val, max_val, median, q25, q75,
                            n_zeros, n_total, sparsity, histogram_data)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        feature_stats_batch,
                    )
                    logger.info(
                        f"Consolidated {len(feature_stats_batch)} items into 'feature_stats'."
                    )

                if logit_influences_batch:
                    conn.executemany(
                        """INSERT OR REPLACE INTO logit_influences
                           (feature_id, token_id, token_name, influence, rank)
                           VALUES (?, ?, ?, ?, ?)""",
                        logit_influences_batch,
                    )
                    logger.info(
                        f"Consolidated {len(logit_influences_batch)} items into 'logit_influences'."
                    )

                if top_examples_batch:
                    conn.executemany(
                        """INSERT OR REPLACE INTO top_examples
                           (feature_id, example_id, rank, activation_value)
                           VALUES (?, ?, ?, ?)""",
                        top_examples_batch,
                    )
                    logger.info(
                        f"Consolidated {len(top_examples_batch)} items into 'top_examples'."
                    )
                conn.commit()
        except Exception as e:
            logger.error(
                f"Error during batch insertion of analytics data: {e}",
                exc_info=True,
            )

    else:
        logger.warning(
            f"No 'features' list found in precomputed analytics data from {analytics_data_path}, or data not loaded."
        )
        if not analytics_data_path:
            logger.warning(
                "Path to precomputed_analytics_path was not provided."
            )

    # Note: Example data (examples table) and raw activations (activations table)
    # are assumed to be populated by other commands like 'generate-activations' (for HDF5)
    # or 'streaming-consolidate stream-to-db' (for direct DB population).
    # This command focuses on analytics derived *from* those primary data sources.
    # If --input-activations-path or other raw data paths were provided, logic to populate
    # 'examples' and 'activations' tables would go here, but that overlaps with other commands.

    logger.info(f"Consolidation to database {db_path} complete.")
    try:
        logger.info("Optimizing database...")
        db.vacuum()
    except Exception as e:
        logger.error(f"Failed to optimize database: {e}", exc_info=True)


# Example CLI usage (to be integrated into main cli.py)
# def setup_consolidate_to_db_parser(subparsers):
#     p = subparsers.add_parser('consolidate-to-db', help='Consolidate precomputed analytics into the SQLite database.')
#     p.add_argument('--database-path', type=str, required=True, help='Path to the SQLite database file to create or update.')
#     p.add_argument('--input-correlations-path', type=str, help='Path to JSON file with feature correlations (output of compute-correlations).')
#     p.add_argument('--input-precomputed-analytics-path', type=str, help='Path to .joblib file with consolidated precomputed analytics (output of precompute-analytics).')
#     # Add more arguments here if other data sources are needed, e.g.:
#     # p.add_argument('--input-neuron-summaries-dir', type=str, help='Path to directory with per-neuron JSONs (output of extract-top-examples), if choosing this over precomputed_analytics.')
#     p.set_defaults(func=consolidate_to_db_command)
