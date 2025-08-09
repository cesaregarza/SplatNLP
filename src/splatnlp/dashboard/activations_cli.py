#!/usr/bin/env python3
"""CLI interface for activation generation commands."""

import argparse
import logging
import sys
from pathlib import Path

# Import the actual command function
from splatnlp.dashboard.commands.generate_activations_cmd import (
    generate_activations_command,
)

logger = logging.getLogger(__name__)


def info_command(args):
    """Display information about an existing activation cache."""
    import pickle

    import joblib

    cache_path = Path(args.cache_path)

    if not cache_path.exists():
        print(f"No cache found at: {cache_path}")
        return

    try:
        # Try to load as pickle first (for streaming format)
        with open(cache_path, "rb") as f:
            cache_info = pickle.load(f)
            if "_loader_version" in cache_info:
                print(f"Cache format: {cache_info['_loader_version']}")
                print(f"Shape: {cache_info.get('shape', 'Unknown')}")
                print(
                    f"Metadata path: {cache_info.get('metadata_path', 'Unknown')}"
                )
                print(f"H5 path: {cache_info.get('h5_path', 'None')}")

                # Try to load metadata
                if "metadata_path" in cache_info:
                    metadata_path = Path(cache_info["metadata_path"])
                    if metadata_path.exists():
                        metadata = joblib.load(metadata_path)
                        if "metadata" in metadata:
                            meta = metadata["metadata"]
                            print(f"\nCache statistics:")
                            print(
                                f"  Regular examples: {meta.get('num_regular', 0)}"
                            )
                            print(
                                f"  Null token examples: {meta.get('num_null', 0)}"
                            )
                            print(
                                f"  Total examples: {meta.get('num_regular', 0) + meta.get('num_null', 0)}"
                            )
                            print(
                                f"  Fraction of data used: {meta.get('fraction', 'Unknown')}"
                            )
                            print(
                                f"  Features dimension: {meta.get('sae_output_dim', 'Unknown')}"
                            )
                return
    except:
        pass

    # Try to load as joblib (for legacy format)
    try:
        cache_data = joblib.load(cache_path)
        if isinstance(cache_data, dict):
            print("Cache format: Legacy joblib")
            if "metadata" in cache_data:
                meta = cache_data["metadata"]
                print(f"\nCache statistics:")
                print(f"  Regular examples: {meta.get('num_regular', 0)}")
                print(f"  Null token examples: {meta.get('num_null', 0)}")
                print(
                    f"  Total examples: {meta.get('num_regular', 0) + meta.get('num_null', 0)}"
                )
                print(
                    f"  Fraction of data used: {meta.get('fraction', 'Unknown')}"
                )

            if "all_sae_hidden_activations" in cache_data:
                acts = cache_data["all_sae_hidden_activations"]
                print(f"  Activation shape: {acts.shape}")
    except Exception as e:
        print(f"Error loading cache: {e}")


def main():
    """Main entry point for activations CLI."""
    parser = argparse.ArgumentParser(
        description="SplatNLP Activation Generation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate activation cache"
    )
    generate_parser.add_argument(
        "--primary-model",
        required=True,
        dest="primary_model_checkpoint",
        help="Path to primary model checkpoint",
    )
    generate_parser.add_argument(
        "--sae-model",
        required=True,
        dest="sae_model_checkpoint",
        help="Path to SAE model checkpoint",
    )
    generate_parser.add_argument(
        "--vocab",
        required=True,
        dest="vocab_path",
        help="Path to vocabulary JSON",
    )
    generate_parser.add_argument(
        "--weapon-vocab",
        required=True,
        dest="weapon_vocab_path",
        help="Path to weapon vocabulary JSON",
    )
    generate_parser.add_argument(
        "--data",
        required=True,
        dest="data_path",
        help="Path to tokenized data CSV",
    )
    generate_parser.add_argument(
        "--output",
        required=True,
        dest="output_path",
        help="Output path for activation cache",
    )
    generate_parser.add_argument(
        "--fraction", type=float, default=0.25, help="Fraction of data to use"
    )
    generate_parser.add_argument(
        "--chunk-size",
        type=float,
        default=0.005,
        help="Chunk size for processing",
    )
    generate_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU processing",
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists",
    )
    generate_parser.add_argument(
        "--num-instances-per-set",
        type=int,
        default=1,
        help="Number of instances per set (20 for Ultra, 1 for Full)",
    )

    # Add model architecture parameters with defaults
    generate_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Model embedding dimension (auto-detected if not specified)",
    )
    generate_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Model hidden dimension (auto-detected if not specified)",
    )
    generate_parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers (auto-detected if not specified)",
    )
    generate_parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads (auto-detected if not specified)",
    )
    generate_parser.add_argument(
        "--num-inducing-points",
        type=int,
        default=None,
        help="Number of inducing points (auto-detected if not specified)",
    )
    generate_parser.add_argument(
        "--hook-target",
        default="masked_mean",
        help="Hook target for activation extraction",
    )
    generate_parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for sampling"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show information about activation cache"
    )
    info_parser.add_argument("cache_path", help="Path to activation cache")

    args = parser.parse_args()

    if args.command == "generate":
        generate_activations_command(args)
    elif args.command == "info":
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
