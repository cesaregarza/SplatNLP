"""
Command for loading and processing neuron activation data from the new format.
This includes pair_df.csv, single_token_df.csv, triple_df.csv, acts.npy, and idxs.npy.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from splatnlp.dashboard.database_manager import DuckDBDatabase

logger = logging.getLogger(__name__)


def load_neuron_data_command(args: argparse.Namespace):
    """Main command function for loading neuron data."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load metadata first
    metadata_path = Path(
        "/mnt/e/activations2/outputs/activations.metadata.joblib"
    )
    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}")
        return

    logger.info(f"Loading metadata from {metadata_path}")
    metadata = joblib.load(metadata_path)
    analysis_df = metadata.get("analysis_df_records")
    if analysis_df is None:
        logger.error("analysis_df_records not found in metadata")
        return

    # Initialize database
    db = DuckDBDatabase(args.database_path)

    # Load and process neuron data
    neuron_path = Path("/mnt/e/activations2/outputs/neuron_acts/neuron_0000")

    # Load CSV files
    pair_df = pd.read_csv(neuron_path / "pair_df.csv")
    single_token_df = pd.read_csv(neuron_path / "single_token_df.csv")
    triple_df = pd.read_csv(neuron_path / "triple_df.csv")

    # Load numpy arrays
    acts = np.load(neuron_path / "acts.npy")
    idxs = np.load(neuron_path / "idxs.npy")

    # Store data in database
    logger.info("Storing data in database...")

    # Create tables if they don't exist
    db.conn.execute(
        """
    CREATE TABLE IF NOT EXISTS neuron_pairs (
        id INTEGER PRIMARY KEY,
        token1 TEXT,
        token2 TEXT,
        activation FLOAT,
        weapon_id INTEGER
    )
    """
    )

    db.conn.execute(
        """
    CREATE TABLE IF NOT EXISTS neuron_single_tokens (
        id INTEGER PRIMARY KEY,
        token TEXT,
        activation FLOAT,
        weapon_id INTEGER
    )
    """
    )

    db.conn.execute(
        """
    CREATE TABLE IF NOT EXISTS neuron_triples (
        id INTEGER PRIMARY KEY,
        token1 TEXT,
        token2 TEXT,
        token3 TEXT,
        activation FLOAT,
        weapon_id INTEGER
    )
    """
    )

    db.conn.execute(
        """
    CREATE TABLE IF NOT EXISTS neuron_activations (
        id INTEGER PRIMARY KEY,
        activation FLOAT,
        example_idx INTEGER
    )
    """
    )

    # Insert data
    db.insert_dataframe("neuron_pairs", pair_df)
    db.insert_dataframe("neuron_single_tokens", single_token_df)
    db.insert_dataframe("neuron_triples", triple_df)

    # Store activations and indices
    activations_df = pd.DataFrame({"activation": acts, "example_idx": idxs})
    db.insert_dataframe("neuron_activations", activations_df)

    # Store analysis dataframe
    db.insert_dataframe("analysis_data", analysis_df)

    logger.info("Data loading complete!")


def register_load_neuron_data_parser(subparsers):
    """Register the load-neuron-data command parser."""
    parser = subparsers.add_parser(
        "load-neuron-data",
        help="Load neuron activation data from new format into database.",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        required=True,
        help="Path to the DuckDB database file",
    )
    parser.set_defaults(func=load_neuron_data_command)
