#!/usr/bin/env python3
"""Command-line interface for the SplatNLP SAE Feature Dashboard."""

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import orjson
import pandas as pd
import torch

from splatnlp.dashboard.app import DASHBOARD_CONTEXT as app_context_ref
from splatnlp.dashboard.app import app

# Import available command functions
from splatnlp.dashboard.commands.generate_activations_cmd import (
    generate_activations_command,
)
from splatnlp.dashboard.components.feature_labels import FeatureLabelsManager
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dashboard_data(args_ns: argparse.Namespace):
    logger.info("Loading vocabularies...")
    with open(args_ns.vocab_path, "rb") as f:
        vocab = orjson.loads(f.read())
    with open(args_ns.weapon_vocab_path, "rb") as f:
        weapon_vocab = orjson.loads(f.read())
    inv_vocab = {v: k for k, v in vocab.items()}
    inv_weapon_vocab = {v: k for k, v in weapon_vocab.items()}
    pad_token_id = vocab.get("<PAD>")

    primary_model = None
    sae_model = None
    feature_clusters = None
    if args_ns.enable_dynamic_tooltips:
        logger.info("Loading models for dynamic tooltips...")
        primary_model = SetCompletionModel(
            vocab_size=len(vocab),
            weapon_vocab_size=len(weapon_vocab),
            embedding_dim=args_ns.primary_embedding_dim,
            hidden_dim=args_ns.primary_hidden_dim,
            output_dim=len(vocab),
            num_layers=args_ns.primary_num_layers,
            num_heads=args_ns.primary_num_heads,
            num_inducing_points=args_ns.primary_num_inducing,
            use_layer_norm=True,
            dropout=0.3,
            pad_token_id=pad_token_id,
        )
        primary_model.load_state_dict(
            torch.load(
                args_ns.primary_model_checkpoint,
                map_location=torch.device("cuda"),
            )
        )
        primary_model.to(DEVICE).eval()

        sae_model = SparseAutoencoder(
            input_dim=args_ns.primary_hidden_dim,
            expansion_factor=args_ns.sae_expansion_factor,
            l1_coefficient=0.001,
            target_usage=0.0,
            usage_coeff=0.0,
        )
        sae_model.load_state_dict(
            torch.load(
                args_ns.sae_model_checkpoint, map_location=torch.device("cuda")
            )
        )
        sae_model.to(DEVICE).eval()

    precomputed_analytics = None  # Will remain None for 'run' command
    db_context = None

    # Handle database initialization based on type
    if args_ns.main_command == "run":
        if args_ns.use_filesystem:
            # Filesystem database case
            logger.info(
                f"Initializing filesystem database with meta_path={args_ns.meta_path} and neurons_root={args_ns.neurons_root}"
            )
            try:
                # Check if we should use cached database with precomputed data
                if hasattr(args_ns, "use_cached_db") and args_ns.use_cached_db:
                    from splatnlp.dashboard.cached_fs_database import (
                        CachedFSDatabase,
                    )

                    precomputed_dir = getattr(args_ns, "precomputed_dir", None)
                    influence_data_path = getattr(
                        args_ns, "influence_data_path", None
                    )

                    db_context = CachedFSDatabase(
                        args_ns.meta_path,
                        args_ns.neurons_root,
                        precomputed_dir=precomputed_dir,
                        influence_data_path=influence_data_path,
                    )
                    logger.info(
                        "Cached filesystem database initialized with precomputed data"
                    )
                else:
                    from splatnlp.dashboard.fs_database import FSDatabase

                    db_context = FSDatabase(
                        args_ns.meta_path, args_ns.neurons_root
                    )
                    logger.info("Filesystem database initialized successfully")
            except Exception as e:
                logger.critical(
                    f"Failed to initialize filesystem database: {e}",
                    exc_info=True,
                )
                raise ConnectionError(
                    f"Critical: Filesystem database could not be initialized: {e}"
                ) from e
        elif args_ns.use_efficient:
            # Efficient database case
            logger.info(
                f"Initializing efficient database with data_dir={args_ns.data_dir} and examples_dir={args_ns.examples_dir}"
            )
            try:
                # Check if we should use cached efficient database with influence data
                influence_data_path = getattr(
                    args_ns, "influence_data_path", None
                )
                precomputed_dir = getattr(args_ns, "precomputed_dir", None)

                if influence_data_path or precomputed_dir:
                    from splatnlp.dashboard.cached_efficient_database import (
                        CachedEfficientDatabase,
                    )

                    db_context = CachedEfficientDatabase(
                        args_ns.data_dir,
                        args_ns.examples_dir,
                        influence_data_path=influence_data_path,
                        precomputed_dir=precomputed_dir,
                    )
                    logger.info(
                        "Cached efficient database initialized with precomputed data"
                    )
                else:
                    from splatnlp.dashboard.efficient_fs_database import (
                        EfficientFSDatabase,
                    )

                    db_context = EfficientFSDatabase(
                        args_ns.data_dir, args_ns.examples_dir
                    )
                    logger.info("Efficient database initialized successfully")
            except Exception as e:
                logger.critical(
                    f"Failed to initialize efficient database: {e}",
                    exc_info=True,
                )
                raise ConnectionError(
                    f"Critical: Efficient database could not be initialized: {e}"
                ) from e

    # Fallback to precomputed_analytics for commands other than 'run' if they support it
    if (
        db_context is None
        and args_ns.main_command != "run"
        and hasattr(args_ns, "precomputed_analytics_path")
        and args_ns.precomputed_analytics_path
    ):
        analytics_path = args_ns.precomputed_analytics_path
        if os.path.exists(analytics_path):
            logger.info(
                f"Loading precomputed analytics from {analytics_path} for command {args_ns.main_command}"
            )
            try:
                precomputed_analytics = joblib.load(analytics_path)
                logger.info(
                    f"Successfully loaded precomputed analytics. Found data for {len(precomputed_analytics.get('features', []))} features."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load precomputed analytics file {analytics_path}: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"Precomputed analytics path {analytics_path} specified but not found."
            )

    # Final check based on the command being run
    if args_ns.main_command == "run" and db_context is None:
        # This implies the DB loading failed and the error wasn't caught or re-raised as expected.
        # Or, the logic path is flawed. This is a safeguard.
        logger.critical(
            "CRITICAL: db_context is None for 'run' command despite database initialization being required. Exiting."
        )
        sys.exit(1)  # Exit if run command has no DB context
    elif (
        args_ns.main_command != "run"
        and db_context is None
        and precomputed_analytics is None
    ):
        logger.warning(
            f"For command '{args_ns.main_command}', no database or precomputed analytics available. Command may not function as expected."
        )

    # Optional feature clusters (for visualization)
    clusters_path = getattr(args_ns, "feature_clusters_path", None)
    if clusters_path:
        cluster_path_obj = Path(clusters_path)
        if cluster_path_obj.exists():
            try:
                feature_clusters = pd.read_parquet(cluster_path_obj)
                logger.info(
                    "Loaded feature clusters from "
                    f"{cluster_path_obj} ({len(feature_clusters)} rows)."
                )
            except Exception as exc:
                logger.warning(
                    "Could not load feature clusters from "
                    f"{cluster_path_obj}: {exc}"
                )
        else:
            logger.warning(
                f"Feature clusters path not found: {cluster_path_obj}"
            )

    # Get model type from args, default to "full" if not specified
    model_type = getattr(args_ns, "model_type", "full")
    feature_labels_manager = FeatureLabelsManager(model_type=model_type)

    dashboard_context_data = SimpleNamespace(
        vocab=vocab,
        inv_vocab=inv_vocab,
        weapon_vocab=weapon_vocab,
        inv_weapon_vocab=inv_weapon_vocab,
        primary_model=primary_model,
        sae_model=sae_model,
        precomputed_analytics=precomputed_analytics,
        db_context=db_context,
        feature_labels_manager=feature_labels_manager,
        device=DEVICE,
        model_type=model_type,
        feature_clusters=feature_clusters,
    )
    return dashboard_context_data


def setup_run_parser(subparsers):
    run_parser = subparsers.add_parser("run", help="Run the dashboard server.")

    # Model type selection
    run_parser.add_argument(
        "--model-type",
        type=str,
        choices=["full", "ultra"],
        default="full",
        help="Model type to use (full: 2048 features, ultra: 24576 features)",
    )

    # Database type selection
    db_group = run_parser.add_mutually_exclusive_group(required=True)
    db_group.add_argument(
        "--database-path",
        type=str,
        help="Path to dashboard database (DuckDB file)",
    )
    db_group.add_argument(
        "--use-filesystem",
        action="store_true",
        help="Use filesystem-based database instead of DuckDB",
    )
    db_group.add_argument(
        "--use-efficient",
        action="store_true",
        help="Use efficient optimized storage (recommended for Ultra model)",
    )

    # Filesystem database specific args
    run_parser.add_argument(
        "--meta-path",
        type=str,
        help="Path to metadata joblib file (required if --use-filesystem)",
    )
    run_parser.add_argument(
        "--neurons-root",
        type=str,
        help="Path to root directory containing neuron_XXXX folders (required if --use-filesystem)",
    )
    # Cached database options
    run_parser.add_argument(
        "--use-cached-db",
        action="store_true",
        help="Use cached database with precomputed data for faster startup",
    )
    run_parser.add_argument(
        "--precomputed-dir",
        type=str,
        help="Directory containing precomputed histogram and stats data",
    )
    run_parser.add_argument(
        "--influence-data-path",
        type=str,
        help="Path to precomputed influence data (CSV or Parquet)",
    )

    # Efficient database specific args
    run_parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to Parquet/Zarr converted data (for --use-efficient). Defaults based on model-type.",
    )
    run_parser.add_argument(
        "--examples-dir",
        type=str,
        help="Path to optimized examples storage (for --use-efficient). Defaults based on model-type.",
    )

    run_parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to vocabulary JSON file",
    )
    run_parser.add_argument(
        "--weapon-vocab-path",
        type=str,
        required=True,
        help="Path to weapon vocabulary JSON file",
    )

    run_parser.add_argument(
        "--enable-dynamic-tooltips",
        action="store_true",
        help="Enable dynamic tooltip computation (requires model checkpoints)",
    )
    run_parser.add_argument(
        "--primary-model-checkpoint",
        type=str,
        help="Path to primary model checkpoint (required if --enable-dynamic-tooltips)",
    )
    run_parser.add_argument(
        "--sae-model-checkpoint",
        type=str,
        help="Path to SAE model checkpoint (required if --enable-dynamic-tooltips)",
    )

    run_parser.add_argument("--primary-embedding-dim", type=int, default=32)
    run_parser.add_argument("--primary-hidden-dim", type=int, default=512)
    run_parser.add_argument("--primary-num-layers", type=int, default=3)
    run_parser.add_argument("--primary-num-heads", type=int, default=8)
    run_parser.add_argument("--primary-num-inducing", type=int, default=32)
    run_parser.add_argument("--sae-expansion-factor", type=float, default=4.0)

    run_parser.add_argument(
        "--feature-clusters-path",
        type=str,
        help="Path to feature cluster parquet for visualization.",
    )
    run_parser.add_argument("--host", type=str, default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=8050)
    run_parser.add_argument("--debug", action="store_true")
    run_parser.set_defaults(func=run_dashboard_server)


def convert_to_efficient_command(args):
    """Run the convert to efficient format command."""
    from splatnlp.dashboard.commands.convert_to_efficient_cmd import (
        BatchConverter,
    )

    converter = BatchConverter(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        chunk_size=args.chunk_size,
        compression=args.compression,
        compression_level=args.compression_level,
    )
    converter.convert_all()


def create_examples_storage_command(args):
    """Run the create examples storage command."""
    from splatnlp.dashboard.commands.create_examples_storage_cmd import (
        OptimizedExampleStorage,
    )

    builder = OptimizedExampleStorage(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        activation_threshold=args.threshold,
        n_features=args.n_features,
    )
    builder.process_all_batches()


def analyze_activations_command(args):
    """Run the analyze activations command."""
    from splatnlp.dashboard.commands.analyze_activations_cmd import (
        ActivationAnalyzer,
    )

    analyzer = ActivationAnalyzer(args.data_dir)

    if args.feature_id is not None:
        # Analyze specific feature
        examples = analyzer.find_top_activated_features(n_features=1)
        print(f"Analysis for feature {args.feature_id}:")
        print(examples)
    elif args.weapon_id is not None:
        # Analyze weapon patterns
        analysis = analyzer.analyze_weapon_patterns(
            args.weapon_id, limit=args.limit
        )
        print(f"Analysis for weapon {args.weapon_id}:")
        print(analysis)
    else:
        # General analysis
        top_features = analyzer.find_top_activated_features(n_features=10)
        print("Top 10 activated features:")
        for feat in top_features:
            print(
                f"  Feature {feat['feature_id']}: {feat['activation_count']} activations"
            )


def run_dashboard_server(args):
    # Set defaults based on model type
    if args.use_efficient:
        if args.model_type == "ultra":
            if not args.data_dir:
                args.data_dir = "/mnt/e/activations_ultra_efficient"
            if not args.examples_dir:
                args.examples_dir = "/mnt/e/dashboard_examples_optimized"
        else:  # full model
            if not args.data_dir:
                args.data_dir = "/mnt/e/activations_full_efficient"
            if not args.examples_dir:
                args.examples_dir = "/mnt/e/dashboard_examples_full_optimized"

    if args.enable_dynamic_tooltips:
        if not args.primary_model_checkpoint or not args.sae_model_checkpoint:
            logger.error(
                "--primary-model-checkpoint and --sae-model-checkpoint are required when --enable-dynamic-tooltips is set"
            )
            sys.exit(1)

    # Validate filesystem database args if using filesystem
    if args.use_filesystem:
        if not args.meta_path or not args.neurons_root:
            logger.error(
                "--meta-path and --neurons-root are required when --use-filesystem is set"
            )
            sys.exit(1)

    dashboard_data_obj = load_dashboard_data(args)
    app_context_ref.vocab = dashboard_data_obj.vocab
    app_context_ref.inv_vocab = dashboard_data_obj.inv_vocab
    app_context_ref.weapon_vocab = dashboard_data_obj.weapon_vocab
    app_context_ref.inv_weapon_vocab = dashboard_data_obj.inv_weapon_vocab
    app_context_ref.primary_model = dashboard_data_obj.primary_model
    app_context_ref.sae_model = dashboard_data_obj.sae_model
    app_context_ref.precomputed_analytics = (
        None  # For 'run' command, precomputed_analytics is not used.
    )
    app_context_ref.db_context = dashboard_data_obj.db_context
    app_context_ref.db = (
        dashboard_data_obj.db_context
    )  # Also set as 'db' for compatibility
    app_context_ref.feature_labels_manager = (
        dashboard_data_obj.feature_labels_manager
    )
    app_context_ref.device = dashboard_data_obj.device
    app_context_ref.model_type = dashboard_data_obj.model_type
    app_context_ref.feature_clusters = dashboard_data_obj.feature_clusters

    # Set influence data if available
    if hasattr(dashboard_data_obj.db_context, "influence_data"):
        app_context_ref.influence_data = (
            dashboard_data_obj.db_context.influence_data
        )

    # Initialize database based on type
    if args.use_efficient:
        from splatnlp.dashboard.app import init_efficient_database

        init_efficient_database(args.data_dir, args.examples_dir)
    elif args.use_filesystem:
        from splatnlp.dashboard.app import init_filesystem_database

        init_filesystem_database(args.meta_path, args.neurons_root)
    else:
        from splatnlp.dashboard.app import init_duckdb_database

        init_duckdb_database(args.database_path)

    logger.info("Starting Dash dashboard server...")
    app.run(host=args.host, port=args.port, debug=args.debug)


def setup_generate_activations_parser(subparsers):
    p = subparsers.add_parser(
        "generate-activations",
        help="Generate and cache activations for dashboard visualization.",
    )
    p.add_argument(
        "--primary-model-checkpoint",
        type=str,
        required=True,
        help="Path to primary model checkpoint",
    )
    p.add_argument(
        "--sae-model-checkpoint",
        type=str,
        required=True,
        help="Path to SAE model checkpoint",
    )
    p.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to vocabulary JSON file",
    )
    p.add_argument(
        "--weapon-vocab-path",
        type=str,
        required=True,
        help="Path to weapon vocabulary JSON file",
    )
    p.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to tokenized data CSV",
    )
    p.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save activation cache",
    )
    p.add_argument(
        "--fraction", type=float, default=0.1, help="Fraction of data to use"
    )
    p.add_argument(
        "--chunk-size",
        type=float,
        default=0.01,
        help="Fraction of data per chunk for chunked processing",
    )
    p.add_argument(
        "--chunk-storage-dir",
        type=str,
        default="activation_chunks_tmp",
        help="Directory to store temporary chunks if using chunked processing",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data sampling",
    )
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-inducing-points", type=int, default=32)
    p.add_argument("--sae-expansion-factor", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--num-instances-per-set",
        type=int,
        default=1,
        help="Number of subset instances per set (1 for Full model, 20 for Ultra model)",
    )
    p.add_argument(
        "--hook-target",
        type=str,
        default="masked_mean",
        help="Target layer for activation hook",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration if cache exists",
    )
    p.set_defaults(func=generate_activations_command)


# COMMENTED OUT - OLD COMMAND
"""
def setup_extract_top_examples_parser(subparsers):
    p = subparsers.add_parser(
        "extract-top-examples",
        help="Extract top examples per activation range for each neuron.",
    )
    p.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to HDF5 file with activations",
    )
    p.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to pickle file with metadata for examples",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-neuron JSON files",
    )
    p.add_argument(
        "--consolidated-output-file",
        type=str,
        help="Optional path to CSV file for consolidated output",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of activation ranges (bins)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Top K examples to store per bin",
    )
    p.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="Maximum number of neurons to process (optional)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )

    # MODIFIED PART STARTS HERE
    p.add_argument(
        "--no-resume",
        action="store_true",  # If flag is present, args.no_resume becomes True
        help="If passed, disables resume behavior (e.g., overwrites files instead of skipping/appending).",
    )
    # No set_defaults needed for 'no_resume' here, as action="store_true" defaults to False if flag is absent.
    # This means if --no-resume is NOT passed, args.no_resume will be False, correctly indicating resume is active.
    # MODIFIED PART ENDS HERE

    p.add_argument(
        "--low-act-chunk-size",
        type=int,
        default=50000,
        help="Chunk size for finding low activation examples",
    )
    p.add_argument(
        "--low-act-bottom-bins",
        type=int,
        default=2,
        help="Number of bottom bins to define 'low activation'",
    )
    p.set_defaults(func=extract_top_examples_command)
"""


# COMMENTED OUT - OLD COMMAND
"""
def setup_compute_correlations_parser(subparsers):
    p = subparsers.add_parser(
        "compute-correlations",
        help="Compute neuron-to-neuron correlations efficiently.",
    )
    p.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to HDF5 activations file.",
    )
    p.add_argument(
        "--neuron-data-dir",
        type=str,
        required=True,
        help="Directory with per-neuron JSON data (from extract-top-examples).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for correlations.",
    )
    p.add_argument(
        "--n-neurons",
        type=int,
        default=None,
        help="Max number of neurons to process from HDF5.",
    )
    p.add_argument(
        "--top-n-per-bin",
        type=int,
        default=50,
        help="Top N examples from each activation bin to consider for indexing.",
    )
    p.add_argument(
        "--correlation-type",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Type of correlation.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Absolute correlation threshold to report.",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=5,
        help="Minimum number of common top examples for a pair to be a candidate.",
    )
    p.add_argument(
        "--max-pairs-per-neuron",
        type=int,
        default=50,
        help="Max candidate pairs to consider per neuron, sorted by overlap.",
    )
    p.add_argument(
        "--min-common-for-correlation",
        type=int,
        default=3,
        help="Minimum common examples needed to actually compute correlation.",
    )
    p.set_defaults(func=compute_correlations_efficient_command)
"""


# COMMENTED OUT - OLD COMMAND
"""
def setup_consolidate_activations_parser(subparsers):
    p = subparsers.add_parser(
        "consolidate-activations",
        help="Consolidate chunked activations or create dashboard cache.",
    )
    sub_p = p.add_subparsers(
        dest="command", help="Consolidation sub-command", required=True
    )  # 'command' will be on args for consolidate_activations_efficient_command

    # Consolidate .npz chunks to HDF5
    consolidate_cmd_parser = sub_p.add_parser(
        "consolidate",
        help="Consolidate chunked .npz activation files into a single HDF5 file.",
    )
    consolidate_cmd_parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory containing .npz chunk files.",
    )
    consolidate_cmd_parser.add_argument(
        "--output-hdf5",
        type=str,
        required=True,
        help="Output HDF5 file path for consolidated activations.",
    )

    # Create dashboard cache from consolidated HDF5 and other data
    cache_cmd_parser = sub_p.add_parser(
        "create-dashboard-cache",
        help="Create cache files optimized for quick dashboard loading.",
    )
    cache_cmd_parser.add_argument(
        "--consolidated-activations-h5",
        type=str,
        required=True,
        help="Path to the consolidated HDF5 activations file.",
    )
    cache_cmd_parser.add_argument(
        "--consolidated-metadata-pkl",
        type=str,
        required=True,
        help="Path to the consolidated .metadata.pkl file.",
    )
    cache_cmd_parser.add_argument(
        "--neuron-summaries-dir",
        type=str,
        required=True,
        help="Directory containing per-neuron summary JSON files.",
    )
    cache_cmd_parser.add_argument(
        "--dashboard-cache-out-dir",
        type=str,
        required=True,
        help="Output directory where dashboard cache files will be stored.",
    )
    cache_cmd_parser.add_argument(
        "--n-neurons-sample",
        type=int,
        default=100,
        help="Number of neurons to include in the dashboard sample.",
    )
    cache_cmd_parser.add_argument(
        "--n-examples-sample",
        type=int,
        default=50000,
        help="Number of examples to include in the dashboard sample.",
    )
    p.set_defaults(func=consolidate_activations_efficient_command)
"""


# COMMENTED OUT - OLD COMMAND
"""
def setup_precompute_analytics_parser(subparsers):
    p = subparsers.add_parser(
        "precompute-analytics",
        help="Precompute all dashboard analytics for faster loading.",
    )
    p.add_argument(
        "--activations-h5",
        type=str,
        required=True,
        help="Path to HDF5 file with activations.",
    )
    p.add_argument(
        "--metadata-file",
        type=str,
        required=True,
        help="Path to metadata file (e.g., .pkl or .joblib from generate_activations).",
    )
    p.add_argument(
        "--output-analytics-file",
        type=str,
        required=True,
        help="Output file to save precomputed analytics (e.g., .joblib).",
    )
    p.add_argument(
        "--primary-model-checkpoint",
        type=str,
        required=True,
        help="Path to primary model checkpoint (.pt).",
    )
    p.add_argument(
        "--sae-model-checkpoint",
        type=str,
        required=True,
        help="Path to SAE model checkpoint (.pt).",
    )
    p.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to vocabulary JSON file.",
    )
    p.add_argument(
        "--weapon-vocab-path",
        type=str,
        required=True,
        help="Path to weapon vocabulary JSON file.",
    )
    p.add_argument(
        "--pad-token-name",
        type=str,
        default="<PAD>",
        help="Name of the padding token in the vocab.",
    )
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-inducing-points", type=int, default=32)
    p.add_argument("--sae-expansion-factor", type=float, default=4.0)
    p.add_argument("--top-k-examples", type=int, default=20)
    p.add_argument("--n-intervals", type=int, default=10)
    p.add_argument("--examples-per-interval", type=int, default=5)
    p.add_argument("--top-k-logit-tokens", type=int, default=5)
    p.add_argument("--top-k-corr-features", type=int, default=5)
    p.add_argument("--top-k-corr-tokens", type=int, default=10)
    p.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device to use ('cuda', 'cpu').",
    )
    p.set_defaults(func=precompute_analytics_command)
"""


# COMMENTED OUT - OLD COMMAND
"""
def setup_streaming_consolidate_parser(subparsers):
    p = subparsers.add_parser(
        "streaming-consolidate",
        help="Streaming consolidation to DB and streaming analytics computation.",
    )
    sub_p = p.add_subparsers(
        dest="command", help="Streaming sub-command", required=True
    )

    stream_db_parser = sub_p.add_parser(
        "stream-to-db",
        help="Stream activation chunks from files into a database.",
    )
    stream_db_parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory containing cached .npz or .pkl chunk files.",
    )
    stream_db_parser.add_argument(
        "--output-db-path",
        type=str,
        required=True,
        help="Path to the output SQLite database file.",
    )
    stream_db_parser.add_argument("--batch-size", type=int, default=1000)
    stream_db_parser.add_argument(
        "--activation-threshold", type=float, default=1e-6
    )
    stream_db_parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not attempt to resume from existing DB data.",
    )

    compute_db_analytics_parser = sub_p.add_parser(
        "compute-db-analytics",
        help="Compute analytics (stats, top examples) from data in the database.",
    )
    compute_db_analytics_parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the SQLite database file containing activations.",
    )
    compute_db_analytics_parser.add_argument(
        "--features-per-batch", type=int, default=100
    )
    compute_db_analytics_parser.add_argument(
        "--max-features", type=int, default=None
    )
    p.set_defaults(func=streaming_consolidate_command)
"""


def main():
    parser = argparse.ArgumentParser(
        description="SplatNLP SAE Feature Dashboard CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Commands", dest="main_command", help="Available commands"
    )
    # subparsers.required = True # Keep this commented if 'main_command' is added to args by set_defaults in each setup function

    setup_run_parser(subparsers)
    setup_generate_activations_parser(subparsers)

    # Commands that need fixing/removal:
    # setup_extract_top_examples_parser(subparsers)
    # setup_compute_correlations_parser(subparsers)
    # setup_consolidate_activations_parser(subparsers)
    # setup_precompute_analytics_parser(subparsers)
    # setup_streaming_consolidate_parser(subparsers)
    # setup_consolidate_to_db_parser(subparsers)
    # register_ingest_binned_samples_parser(subparsers)
    # setup_minimize_examples_parser(subparsers)

    # Add new efficient commands
    setup_convert_to_efficient_parser(subparsers)
    setup_create_examples_storage_parser(subparsers)
    setup_analyze_activations_parser(subparsers)

    args = parser.parse_args()

    # Ensure that func is present (it should be, due to set_defaults)
    if hasattr(args, "func"):
        # Add main_command to args if it's not already there (e.g. for run command)
        # For other commands that have sub-subcommands, args.command will exist.
        # For commands like 'run', 'generate-activations', etc., args.main_command is set by dest='main_command'.
        if not hasattr(args, "main_command") and hasattr(args, "command"):
            # This might happen if a command has its own subparser named 'command'
            # For clarity, ensure 'main_command' reflects the top-level command.
            # However, the dest='main_command' in add_subparsers should handle this for top-level.
            pass  # args.main_command should be correctly set by the top-level subparser.

        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


def setup_convert_to_efficient_parser(subparsers):
    """Setup parser for converting joblib to efficient format."""
    parser = subparsers.add_parser(
        "convert-to-efficient",
        help="Convert joblib batches to efficient Parquet/Zarr format",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing joblib batch files",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Zarr chunk size for row dimension",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "lz4", "gzip", "blosc", "none"],
        help="Compression algorithm",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Compression level (1-9)",
    )

    parser.set_defaults(func=convert_to_efficient_command)


def setup_create_examples_storage_parser(subparsers):
    """Setup parser for creating optimized example storage."""
    parser = subparsers.add_parser(
        "create-examples",
        help="Create optimized example storage for dashboard",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with Parquet/Zarr converted data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for example storage",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top examples to store per feature",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum activation threshold",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=24576,
        help="Number of features to process",
    )

    parser.set_defaults(func=create_examples_storage_command)


def setup_analyze_activations_parser(subparsers):
    """Setup parser for analyzing activations."""
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze activations using efficient storage",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/e/activations_ultra_efficient",
        help="Directory with converted data",
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        help="Specific feature to analyze",
    )
    parser.add_argument(
        "--weapon-id",
        type=int,
        help="Analyze patterns for specific weapon",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit for number of examples",
    )

    parser.set_defaults(func=analyze_activations_command)


# COMMENTED OUT - OLD COMMAND
"""
def setup_consolidate_to_db_parser(subparsers):
    p = subparsers.add_parser(
        "consolidate-to-db",
        help="Consolidate precomputed analytics from files into the SQLite database.",
    )
    p.add_argument(
        "--database-path",
        type=str,
        required=True,
        help="Path to the SQLite database file to create or update.",
    )
    p.add_argument(
        "--input-correlations-path",
        type=str,
        help="Path to JSON file with feature correlations (e.g., output of compute-correlations).",
    )
    p.add_argument(
        "--input-precomputed-analytics-path",
        type=str,
        help="Path to .joblib file with consolidated precomputed analytics (e.g., output of precompute-analytics). This file is expected to contain feature statistics, top examples, and logit influences.",
    )
    p.set_defaults(func=consolidate_to_db_command)
"""


# Note: register_ingest_binned_samples_parser is defined in ingest_binned_samples_cmd.py
# and called directly in main() above.

if __name__ == "__main__":
    main()
