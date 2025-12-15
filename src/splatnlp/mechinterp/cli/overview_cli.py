"""CLI for feature overview.

Usage:
    # Basic overview
    poetry run python -m splatnlp.mechinterp.cli.overview_cli \
        --feature-id 18712 --model ultra --max-examples 2000

    # Extended analyses
    poetry run python -m splatnlp.mechinterp.cli.overview_cli \
        --feature-id 6235 --model ultra --enrichment --binary --kit

    # All extended analyses
    poetry run python -m splatnlp.mechinterp.cli.overview_cli \
        --feature-id 6235 --model ultra --all
"""

import argparse
import json
import logging
import sys
from typing import Literal

from splatnlp.mechinterp.labeling.overview import (
    BinaryEnrichment,
    KitBreakdown,
    RegionBreakdown,
    TokenEnrichment,
    compute_binary_enrichment,
    compute_kit_breakdown,
    compute_overview,
    compute_region_breakdown,
    compute_token_enrichment,
)
from splatnlp.mechinterp.skill_helpers.context_loader import load_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Markdown Formatters
# =============================================================================


def _enrichment_to_markdown(enrichments: list[TokenEnrichment]) -> str:
    """Format token enrichment as markdown."""
    lines = [
        "## Token Enrichment Analysis",
        "",
        "Tokens enriched (> 1.2x) or suppressed (< 0.8x) in high-activation examples.",
        "",
    ]

    suppressors = [e for e in enrichments if e.is_suppressor]
    enhancers = [e for e in enrichments if e.is_enhancer]

    if suppressors:
        lines.append("### Suppressors (under-represented in high activation)")
        lines.append("")
        lines.append(
            "| Token | Family | Baseline | High | Enrichment |"
        )
        lines.append("|-------|--------|----------|------|------------|")
        for e in suppressors[:15]:
            lines.append(
                f"| {e.token} | {e.family} | {e.baseline_rate:.1%} | "
                f"{e.high_rate:.1%} | **{e.enrichment_ratio:.2f}x** |"
            )
        lines.append("")

    if enhancers:
        lines.append("### Enhancers (over-represented in high activation)")
        lines.append("")
        lines.append(
            "| Token | Family | Baseline | High | Enrichment |"
        )
        lines.append("|-------|--------|----------|------|------------|")
        for e in sorted(enhancers, key=lambda x: -x.enrichment_ratio)[:15]:
            lines.append(
                f"| {e.token} | {e.family} | {e.baseline_rate:.1%} | "
                f"{e.high_rate:.1%} | **{e.enrichment_ratio:.2f}x** |"
            )
        lines.append("")

    if not suppressors and not enhancers:
        lines.append("*No significant enrichment detected.*")
        lines.append("")

    return "\n".join(lines)


def _regions_to_markdown(regions: list[RegionBreakdown]) -> str:
    """Format region breakdown as markdown."""
    lines = [
        "## Activation Region Breakdown",
        "",
        "Anti-flanderization analysis: how does the feature behave across "
        "different activation levels?",
        "",
    ]

    for r in regions:
        if r.n_examples == 0:
            continue

        lines.append(f"### {r.region_name}")
        lines.append(f"- Examples: {r.n_examples} ({r.pct_of_total:.1f}%)")
        lines.append("")

        if r.top_weapons:
            lines.append("**Top Weapons:**")
            for wname, count, pct in r.top_weapons[:5]:
                lines.append(f"- {wname}: {count} ({pct:.1f}%)")
            lines.append("")

        if r.top_families:
            lines.append("**Top Families:**")
            for fname, freq in r.top_families[:5]:
                lines.append(f"- {fname}: {freq:.1%}")
            lines.append("")

    return "\n".join(lines)


def _binary_to_markdown(enrichments: list[BinaryEnrichment]) -> str:
    """Format binary ability enrichment as markdown."""
    lines = [
        "## Binary Ability Enrichment",
        "",
        "Binary abilities (main-only, no AP scaling) - presence rate and "
        "activation comparison.",
        "",
        "| Ability | Baseline | High | Enrichment | "
        "Mean With | Mean Without | Delta |",
        "|---------|----------|------|------------|"
        "----------|--------------|-------|",
    ]

    for e in enrichments:
        delta_fmt = f"+{e.delta:.4f}" if e.delta > 0 else f"{e.delta:.4f}"
        lines.append(
            f"| {e.ability} | {e.baseline_presence:.1%} | "
            f"{e.high_presence:.1%} | **{e.enrichment:.2f}x** | "
            f"{e.mean_with:.4f} | {e.mean_without:.4f} | {delta_fmt} |"
        )

    lines.append("")

    # Highlight significant findings
    suppressed = [e for e in enrichments if e.enrichment < 0.5]
    enhanced = [e for e in enrichments if e.enrichment > 1.5]

    if suppressed:
        lines.append("**Suppressed abilities (< 0.5x):**")
        for e in suppressed:
            lines.append(f"- {e.ability}: {e.enrichment:.2f}x")
        lines.append("")

    if enhanced:
        lines.append("**Enhanced abilities (> 1.5x):**")
        for e in enhanced:
            lines.append(f"- {e.ability}: {e.enrichment:.2f}x")
        lines.append("")

    return "\n".join(lines)


def _kit_to_markdown(breakdown: KitBreakdown) -> str:
    """Format kit breakdown as markdown."""
    lines = [
        f"## Kit Breakdown ({breakdown.region.title()} Region)",
        "",
        f"Sub/special weapons in {breakdown.n_examples} examples.",
        "",
    ]

    if breakdown.subs:
        lines.append("### Sub Weapons")
        lines.append("")
        lines.append("| Sub | Count | % |")
        lines.append("|-----|-------|---|")
        for name, count, pct in breakdown.subs[:10]:
            lines.append(f"| {name} | {count} | {pct:.1f}% |")
        lines.append("")

    if breakdown.specials:
        lines.append("### Special Weapons")
        lines.append("")
        lines.append("| Special | Count | % |")
        lines.append("|---------|-------|---|")
        for name, count, pct in breakdown.specials[:10]:
            lines.append(f"| {name} | {count} | {pct:.1f}% |")
        lines.append("")

    if not breakdown.subs and not breakdown.specials:
        lines.append("*No kit data available.*")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


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

    # Extended analysis flags
    parser.add_argument(
        "--enrichment",
        action="store_true",
        help="Show token enrichment analysis (enhancers/suppressors)",
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Show activation region breakdown (anti-flanderization)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Show binary ability enrichment (main-only abilities)",
    )
    parser.add_argument(
        "--kit",
        action="store_true",
        help="Show sub/special weapon breakdown for core region",
    )
    parser.add_argument(
        "--kit-region",
        type=str,
        choices=["core", "high", "all"],
        default="core",
        help="Region for kit breakdown (default: core)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all extended analyses (--enrichment --regions --binary --kit)",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=0.90,
        help="Percentile threshold for 'high activation' (default: 0.90)",
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        args.enrichment = True
        args.regions = True
        args.binary = True
        args.kit = True

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

        # Extended analyses (markdown only)
        if args.enrichment:
            logger.info("Computing token enrichment...")
            enrichments = compute_token_enrichment(
                args.feature_id, ctx, high_percentile=args.high_percentile
            )
            print(_enrichment_to_markdown(enrichments))

        if args.regions:
            logger.info("Computing region breakdown...")
            regions = compute_region_breakdown(args.feature_id, ctx)
            print(_regions_to_markdown(regions))

        if args.binary:
            logger.info("Computing binary ability enrichment...")
            binary = compute_binary_enrichment(
                args.feature_id, ctx, high_percentile=args.high_percentile
            )
            print(_binary_to_markdown(binary))

        if args.kit:
            logger.info(f"Computing kit breakdown for {args.kit_region} region...")
            kit = compute_kit_breakdown(
                args.feature_id, ctx, region=args.kit_region
            )
            print(_kit_to_markdown(kit))

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

        # Extended analyses in JSON mode
        if args.enrichment:
            enrichments = compute_token_enrichment(
                args.feature_id, ctx, high_percentile=args.high_percentile
            )
            output["token_enrichment"] = [
                {
                    "token": e.token,
                    "family": e.family,
                    "ap_level": e.ap_level,
                    "baseline_rate": e.baseline_rate,
                    "high_rate": e.high_rate,
                    "enrichment_ratio": e.enrichment_ratio,
                    "is_enhancer": e.is_enhancer,
                    "is_suppressor": e.is_suppressor,
                }
                for e in enrichments
            ]

        if args.regions:
            regions = compute_region_breakdown(args.feature_id, ctx)
            output["regions"] = [
                {
                    "region_name": r.region_name,
                    "pct_range": r.pct_range,
                    "n_examples": r.n_examples,
                    "pct_of_total": r.pct_of_total,
                    "top_weapons": r.top_weapons,
                    "top_families": r.top_families,
                }
                for r in regions
            ]

        if args.binary:
            binary = compute_binary_enrichment(
                args.feature_id, ctx, high_percentile=args.high_percentile
            )
            output["binary_enrichment"] = [
                {
                    "ability": e.ability,
                    "baseline_presence": e.baseline_presence,
                    "high_presence": e.high_presence,
                    "enrichment": e.enrichment,
                    "mean_with": e.mean_with,
                    "mean_without": e.mean_without,
                    "delta": e.delta,
                }
                for e in binary
            ]

        if args.kit:
            kit = compute_kit_breakdown(
                args.feature_id, ctx, region=args.kit_region
            )
            output["kit_breakdown"] = {
                "region": kit.region,
                "n_examples": kit.n_examples,
                "subs": kit.subs,
                "specials": kit.specials,
            }

        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
