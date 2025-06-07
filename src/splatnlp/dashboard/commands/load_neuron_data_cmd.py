#!/usr/bin/env python3
"""Command to load neuron data into the database."""

import argparse
import logging
from pathlib import Path

from splatnlp.dashboard.fs_database import FSDatabase

logger = logging.getLogger(__name__)


def register_load_neuron_data_parser(subparsers):
    """Register the load-neuron-data command parser."""
    parser = subparsers.add_parser(
        "load-neuron-data",
        help="Load neuron data into database",
        description="Load neuron data from filesystem into database.",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        required=True,
        help="Path to metadata joblib file",
    )
    parser.add_argument(
        "--neurons-root",
        type=str,
        required=True,
        help="Path to root directory containing neuron_XXXX folders",
    )
    parser.set_defaults(func=load_neuron_data_command)


def load_neuron_data_command(args: argparse.Namespace) -> None:
    """Load neuron data into database.

    Args:
        args: Command line arguments
    """
    logger.info("Loading neuron data into database...")
    db = FSDatabase(args.meta_path, args.neurons_root)
    logger.info("Neuron data loaded successfully")
