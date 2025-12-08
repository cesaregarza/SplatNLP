"""CLI for feature overview.

Usage:
    poetry run python -m splatnlp.mechinterp.cli.overview_cli \
        --feature-id 18712 --model ultra --max-examples 2000
"""

import argparse
import json
import logging
import sys
from typing import Literal

from splatnlp.mechinterp.labeling.overview import compute_overview
from splatnlp.mechinterp.skill_helpers.context_loader import load_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Get a quick overview of an SAE feature"
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        required=True,
        help="SAE feature ID to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type (default: ultra)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top tokens to show (default: 15)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5000,
        help="Maximum number of activations to load (0 for all, default: 5000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading context for {args.model} model...")
    ctx = load_context(args.model)

    logger.info(f"Computing overview for feature {args.feature_id}...")
    max_examples = None if args.max_examples == 0 else args.max_examples
    overview = compute_overview(
        feature_id=args.feature_id,
        ctx=ctx,
        top_k_tokens=args.top_k,
        max_examples=max_examples,
    )

    if args.format == "markdown":
        print(overview.to_markdown())
    else:
        # JSON output
        output = {
            "feature_id": overview.feature_id,
            "model_type": overview.model_type,
            "activation_mean": overview.activation_mean,
            "activation_std": overview.activation_std,
            "activation_median": overview.activation_median,
            "sparsity": overview.sparsity,
            "n_examples": overview.n_examples,
            "top_tokens": overview.top_tokens,
            "family_breakdown": overview.family_breakdown,
            "top_weapons": overview.top_weapons,
            "sample_contexts": [
                {
                    "tokens": ctx.tokens,
                    "weapon": ctx.weapon,
                    "activation": ctx.activation,
                }
                for ctx in overview.sample_contexts
            ],
            "relu_floor_rate": overview.relu_floor_rate,
            "existing_label": overview.existing_label,
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
