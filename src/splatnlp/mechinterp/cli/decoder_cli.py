"""CLI for decoder weight analysis.

Usage:
    # Get output influence for a feature
    poetry run python -m splatnlp.mechinterp.cli.decoder_cli output-influence \
        --feature-id 13934 --model ultra --top-k 15

    # Get decoder weight magnitude percentile
    poetry run python -m splatnlp.mechinterp.cli.decoder_cli weight-percentile \
        --feature-id 13934 --model ultra

    # Find features with similar decoder patterns
    poetry run python -m splatnlp.mechinterp.cli.decoder_cli similar \
        --feature-id 13934 --model ultra --top-k 10
"""

import argparse
import json
import logging
import sys
from typing import Literal

import torch

from splatnlp.mechinterp.experiments.decoder_output import (
    _extract_ap_level,
    _load_model_weights,
    _load_sae_decoder,
    compute_output_contribution,
    get_feature_decoder_vector,
)
from splatnlp.mechinterp.skill_helpers.context_loader import load_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_output_influence(args):
    """Show what tokens this feature promotes/suppresses."""
    logger.info(f"Loading context for {args.model} model...")
    ctx = load_context(args.model)

    logger.info(f"Computing output influence for feature {args.feature_id}...")
    contribution = compute_output_contribution(args.feature_id, args.model)

    # Parse excluded families (if any)
    exclude_families = set()
    if hasattr(args, "exclude_family") and args.exclude_family:
        exclude_families = {f.strip().lower() for f in args.exclude_family}
        logger.info(f"Excluding families: {exclude_families}")

    # Get token names and build results
    vocab_size = contribution.shape[0]
    special_tokens = {"<PAD>", "<MASK>", "<NULL>", "<UNK>"}

    promoted = []
    suppressed = []
    excluded_promoted = []  # Track what was excluded

    for token_id in range(vocab_size):
        token_name = ctx.inv_vocab.get(token_id, f"token_{token_id}")
        if token_name in special_tokens:
            continue

        contrib_val = float(contribution[token_id])
        family, ap_level = _extract_ap_level(token_name)

        entry = {
            "token": token_name,
            "contribution": round(contrib_val, 4),
            "family": family,
            "ap_level": ap_level,
        }

        # Check if family is excluded
        is_excluded = family and family.lower() in exclude_families

        if is_excluded:
            if contrib_val > 0:
                excluded_promoted.append(entry)
            continue  # Skip excluded families

        if contrib_val > 0:
            promoted.append(entry)
        else:
            suppressed.append(entry)

    # Sort
    promoted.sort(key=lambda x: -x["contribution"])
    suppressed.sort(key=lambda x: x["contribution"])
    excluded_promoted.sort(key=lambda x: -x["contribution"])

    top_promoted = promoted[: args.top_k]
    top_suppressed = suppressed[: args.top_k]

    if args.format == "json":
        output = {
            "feature_id": args.feature_id,
            "model": args.model,
            "promoted": top_promoted,
            "suppressed": top_suppressed,
        }
        if exclude_families:
            output["excluded_families"] = list(exclude_families)
            output["excluded_promoted"] = excluded_promoted[: args.top_k]
        print(json.dumps(output, indent=2))
    else:
        # Markdown output
        if exclude_families:
            print(
                f"## Feature {args.feature_id} Conditional Output Influence ({args.model})\n"
            )
            print(
                f"**Excluding families:** {', '.join(sorted(exclude_families))}\n"
            )
        else:
            print(
                f"## Feature {args.feature_id} Output Influence ({args.model})\n"
            )

        print("### Tokens This Feature PROMOTES\n")
        print("| Token | Contribution | Family | AP Level |")
        print("|-------|--------------|--------|----------|")
        for p in top_promoted:
            ap = p["ap_level"] if p["ap_level"] is not None else "binary"
            print(
                f"| {p['token']} | {p['contribution']:+.4f} | {p['family']} | {ap} |"
            )

        print("\n### Tokens This Feature SUPPRESSES\n")
        print("| Token | Contribution | Family | AP Level |")
        print("|-------|--------------|--------|----------|")
        for s in top_suppressed:
            ap = s["ap_level"] if s["ap_level"] is not None else "binary"
            print(
                f"| {s['token']} | {s['contribution']:+.4f} | {s['family']} | {ap} |"
            )

        # Show what was excluded (if any)
        if exclude_families and excluded_promoted:
            print(
                f"\n### Excluded from {', '.join(sorted(exclude_families))} (would have been promoted)\n"
            )
            print("| Token | Contribution | AP Level |")
            print("|-------|--------------|----------|")
            for e in excluded_promoted[:5]:  # Show top 5 excluded
                ap = e["ap_level"] if e["ap_level"] is not None else "binary"
                print(f"| {e['token']} | {e['contribution']:+.4f} | {ap} |")

        # Add interpretation hints
        print("\n### Interpretation")
        if top_promoted and top_suppressed:
            print(
                f"- **Top promoted**: {top_promoted[0]['token']} ({top_promoted[0]['contribution']:+.4f})"
            )
            print(
                f"- **Top suppressed**: {top_suppressed[0]['token']} ({top_suppressed[0]['contribution']:+.4f})"
            )

            # Check for AP level patterns
            low_ap_promoted = [
                p
                for p in top_promoted
                if p["ap_level"] is not None and p["ap_level"] <= 6
            ]
            high_ap_suppressed = [
                s
                for s in top_suppressed
                if s["ap_level"] is not None and s["ap_level"] >= 51
            ]

            if low_ap_promoted and high_ap_suppressed:
                print(
                    "- **Pattern**: Promotes low-AP tokens, suppresses high-AP stacking"
                )

            if exclude_families:
                print(
                    f"\n**Conditional analysis:** With {', '.join(sorted(exclude_families))} excluded, "
                    f"the next best promoted tokens are shown above."
                )


def cmd_weight_percentile(args):
    """Show decoder weight magnitude and percentile."""
    logger.info(f"Loading SAE decoder for {args.model} model...")
    decoder = _load_sae_decoder(args.model)

    # Compute magnitudes for all features
    n_features = decoder.shape[1]
    magnitudes = torch.norm(decoder, p=2, dim=0)

    target_magnitude = float(magnitudes[args.feature_id])

    # Compute percentile
    percentile = (magnitudes < target_magnitude).float().mean() * 100

    if args.format == "json":
        output = {
            "feature_id": args.feature_id,
            "model": args.model,
            "decoder_magnitude": round(target_magnitude, 4),
            "percentile": round(float(percentile), 1),
            "n_features": n_features,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"## Feature {args.feature_id} Decoder Weight ({args.model})\n")
        print(f"- **Magnitude**: {target_magnitude:.4f}")
        print(f"- **Percentile**: {percentile:.1f}%")
        print(f"- **Total features**: {n_features}")

        if percentile > 90:
            print(
                "\n⚠️ **High magnitude**: This feature has strong decoder weights (top 10%)"
            )
        elif percentile < 10:
            print(
                "\n⚠️ **Low magnitude**: This feature has weak decoder weights (bottom 10%)"
            )


def cmd_similar(args):
    """Find features with similar decoder patterns."""
    logger.info(f"Loading SAE decoder for {args.model} model...")
    decoder = _load_sae_decoder(args.model)

    # Get target feature decoder
    target_decoder = decoder[:, args.feature_id]

    # Compute cosine similarity with all features
    n_features = decoder.shape[1]

    # Normalize
    target_norm = target_decoder / (torch.norm(target_decoder) + 1e-10)
    decoder_norms = decoder / (torch.norm(decoder, dim=0, keepdim=True) + 1e-10)

    similarities = torch.matmul(decoder_norms.T, target_norm)

    # Get top-k (excluding self)
    similarities[args.feature_id] = -1  # Exclude self
    top_indices = torch.argsort(similarities, descending=True)[: args.top_k]

    results = []
    for idx in top_indices:
        idx = int(idx)
        sim = float(similarities[idx])
        results.append({"feature_id": idx, "similarity": round(sim, 4)})

    if args.format == "json":
        output = {
            "query_feature_id": args.feature_id,
            "model": args.model,
            "similar_features": results,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"## Features Similar to {args.feature_id} ({args.model})\n")
        print("| Feature ID | Cosine Similarity |")
        print("|------------|-------------------|")
        for r in results:
            print(f"| {r['feature_id']} | {r['similarity']:.4f} |")


def main():
    parser = argparse.ArgumentParser(
        description="Decoder weight analysis for SAE features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # output-influence subcommand
    p_output = subparsers.add_parser(
        "output-influence",
        help="Show what tokens this feature promotes/suppresses",
    )
    p_output.add_argument(
        "--feature-id",
        type=int,
        required=True,
        help="SAE feature ID",
    )
    p_output.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type (default: ultra)",
    )
    p_output.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top tokens to show (default: 15)",
    )
    p_output.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p_output.add_argument(
        "--exclude-family",
        type=str,
        nargs="+",
        default=None,
        help="Family names to exclude (e.g., --exclude-family swim_speed_up bomb_resistance_up)",
    )
    p_output.set_defaults(func=cmd_output_influence)

    # weight-percentile subcommand
    p_weight = subparsers.add_parser(
        "weight-percentile",
        help="Show decoder weight magnitude and percentile",
    )
    p_weight.add_argument(
        "--feature-id",
        type=int,
        required=True,
        help="SAE feature ID",
    )
    p_weight.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type (default: ultra)",
    )
    p_weight.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p_weight.set_defaults(func=cmd_weight_percentile)

    # similar subcommand
    p_similar = subparsers.add_parser(
        "similar",
        help="Find features with similar decoder patterns",
    )
    p_similar.add_argument(
        "--feature-id",
        type=int,
        required=True,
        help="SAE feature ID",
    )
    p_similar.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type (default: ultra)",
    )
    p_similar.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of similar features to show (default: 10)",
    )
    p_similar.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p_similar.set_defaults(func=cmd_similar)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
