"""CLI entry point for the mechinterp-runner skill.

This module provides the command-line interface for executing
experiment specifications.

Usage (JSON spec mode):
    poetry run python -m splatnlp.mechinterp.cli.runner_cli \\
        --spec-path /mnt/e/mechinterp_runs/specs/my_spec.json

Usage (subcommand mode):
    poetry run python -m splatnlp.mechinterp.cli.runner_cli family-sweep \\
        --feature-id 6235 --family swim_speed_up --model ultra

    poetry run python -m splatnlp.mechinterp.cli.runner_cli heatmap \\
        --feature-id 6235 --family-x quick_respawn --family-y ink_saver_sub

    poetry run python -m splatnlp.mechinterp.cli.runner_cli weapon-sweep \\
        --feature-id 6235 --model ultra --top-k 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from splatnlp.mechinterp.experiments.base import (
    get_runner_for_type,
    list_available_runners,
)
from splatnlp.mechinterp.schemas.experiment_specs import (
    ExperimentSpec,
    ExperimentType,
)
from splatnlp.mechinterp.skill_helpers.context_loader import load_context
from splatnlp.mechinterp.state.io import RESULTS_DIR, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================


def _parse_rungs(rungs_str: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated rungs string into list of ints."""
    if not rungs_str:
        return None
    return [int(r.strip()) for r in rungs_str.split(",")]


def _execute_and_output(spec: ExperimentSpec, args) -> int:
    """Execute experiment spec and output results."""
    # Ensure output directory exists
    ensure_dirs()
    output_dir = getattr(args, "output_dir", RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load context
    try:
        logger.info(f"Loading context for {spec.model_type} model...")
        ctx = load_context(model_type=spec.model_type)
    except Exception as e:
        logger.error(f"Failed to load context: {e}")
        return 1

    # Get runner
    try:
        runner = get_runner_for_type(spec.type)
        logger.info(f"Using runner: {runner.name}")
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Execute experiment
    logger.info("Executing experiment...")
    result = runner.run(spec, ctx)

    # Output based on format
    output_format = getattr(args, "format", "markdown")

    if output_format == "json":
        output_path = output_dir / result.to_filename()
        try:
            with open(output_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"Result written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write result: {e}")
            return 1

    # Print summary
    print("\n" + "=" * 60)
    print(result.get_summary())
    print("=" * 60)

    if output_format == "json":
        print(f"\nResult saved to: {output_path}")

    return 0 if result.success else 1


# =============================================================================
# Subcommand Handlers
# =============================================================================


def cmd_family_sweep(args) -> int:
    """Execute family 1D sweep experiment."""
    spec = ExperimentSpec(
        type=ExperimentType.FAMILY_1D_SWEEP,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "family": args.family,
            "rungs": _parse_rungs(getattr(args, "rungs", None)),
            "include_absent": getattr(args, "include_absent", True),
        },
    )
    return _execute_and_output(spec, args)


def cmd_heatmap(args) -> int:
    """Execute family 2D heatmap experiment."""
    spec = ExperimentSpec(
        type=ExperimentType.FAMILY_2D_HEATMAP,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "family_x": args.family_x,
            "family_y": args.family_y,
            "rungs_x": _parse_rungs(getattr(args, "rungs_x", None)),
            "rungs_y": _parse_rungs(getattr(args, "rungs_y", None)),
        },
    )
    return _execute_and_output(spec, args)


def cmd_weapon_sweep(args) -> int:
    """Execute weapon sweep experiment."""
    spec = ExperimentSpec(
        type=ExperimentType.WEAPON_SWEEP,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "top_k_weapons": getattr(args, "top_k", 20),
            "min_examples": getattr(args, "min_examples", 10),
            "condition_family": getattr(args, "condition_family", None),
        },
    )
    return _execute_and_output(spec, args)


def cmd_kit_sweep(args) -> int:
    """Execute kit sweep experiment (sub/special analysis)."""
    spec = ExperimentSpec(
        type=ExperimentType.KIT_SWEEP,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "top_k": getattr(args, "top_k", 15),
            "min_examples": getattr(args, "min_examples", 10),
            "analyze_combinations": getattr(args, "combinations", False),
        },
    )
    return _execute_and_output(spec, args)


def cmd_binary(args) -> int:
    """Execute binary presence effect experiment."""
    # Parse binary tokens if provided
    binary_tokens = None
    if getattr(args, "tokens", None):
        binary_tokens = [t.strip() for t in args.tokens.split(",")]

    spec = ExperimentSpec(
        type=ExperimentType.BINARY_PRESENCE_EFFECT,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "binary_tokens": binary_tokens,
            "primary_family": getattr(args, "primary_family", None),
            "min_samples": getattr(args, "min_samples", 20),
            "high_percentile": getattr(args, "high_percentile", 90),
        },
    )
    return _execute_and_output(spec, args)


def cmd_coverage(args) -> int:
    """Execute core coverage analysis experiment."""
    # Parse tokens if provided
    tokens = None
    if getattr(args, "tokens", None):
        tokens = [t.strip() for t in args.tokens.split(",")]

    spec = ExperimentSpec(
        type=ExperimentType.CORE_COVERAGE_ANALYSIS,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "tokens_to_check": tokens,
            "coverage_threshold": getattr(args, "threshold", 0.30),
            "include_weapons": getattr(args, "include_weapons", True),
        },
    )
    return _execute_and_output(spec, args)


def cmd_token_influence(args) -> int:
    """Execute token influence sweep experiment."""
    spec = ExperimentSpec(
        type=ExperimentType.TOKEN_INFLUENCE_SWEEP,
        feature_id=args.feature_id,
        model_type=args.model,
        variables={
            "min_samples": getattr(args, "min_samples", 50),
            "high_percentile": getattr(args, "high_percentile", 90),
            "collapse_families": getattr(args, "collapse_families", True),
        },
    )
    return _execute_and_output(spec, args)


# =============================================================================
# Subcommand Parser Setup
# =============================================================================


def _add_common_args(parser):
    """Add common arguments to a subcommand parser."""
    parser.add_argument(
        "--feature-id", "-f",
        type=int,
        required=True,
        help="SAE feature ID to analyze",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type (default: ultra)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write results",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )


def _add_family_sweep_parser(subparsers):
    """Add family-sweep subcommand."""
    p = subparsers.add_parser(
        "family-sweep",
        help="Run family 1D sweep (activation across AP rungs)",
        description="Test how a single ability family affects feature activation",
    )
    _add_common_args(p)
    p.add_argument(
        "--family",
        type=str,
        required=True,
        help="Ability family (e.g., swim_speed_up, quick_respawn)",
    )
    p.add_argument(
        "--rungs",
        type=str,
        default=None,
        help="Comma-separated AP rungs (e.g., 3,6,12,21,29). Default: standard rungs",
    )
    p.add_argument(
        "--no-absent",
        dest="include_absent",
        action="store_false",
        default=True,
        help="Exclude baseline (family absent) from analysis",
    )
    p.set_defaults(func=cmd_family_sweep)


def _add_heatmap_parser(subparsers):
    """Add heatmap subcommand."""
    p = subparsers.add_parser(
        "heatmap",
        help="Run family 2D heatmap (interaction between two families)",
        description="Test interaction between two ability families",
    )
    _add_common_args(p)
    p.add_argument(
        "--family-x",
        type=str,
        required=True,
        help="First ability family (X axis)",
    )
    p.add_argument(
        "--family-y",
        type=str,
        required=True,
        help="Second ability family (Y axis)",
    )
    p.add_argument(
        "--rungs-x",
        type=str,
        default=None,
        help="Comma-separated AP rungs for X axis",
    )
    p.add_argument(
        "--rungs-y",
        type=str,
        default=None,
        help="Comma-separated AP rungs for Y axis",
    )
    p.set_defaults(func=cmd_heatmap)


def _add_weapon_sweep_parser(subparsers):
    """Add weapon-sweep subcommand."""
    p = subparsers.add_parser(
        "weapon-sweep",
        help="Run weapon sweep (activation by weapon)",
        description="Analyze feature activation across different weapons",
    )
    _add_common_args(p)
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top weapons to show (default: 20)",
    )
    p.add_argument(
        "--min-examples",
        type=int,
        default=10,
        help="Minimum examples per weapon (default: 10)",
    )
    p.add_argument(
        "--condition-family",
        type=str,
        default=None,
        help="Only analyze examples with this family present",
    )
    p.set_defaults(func=cmd_weapon_sweep)


def _add_kit_sweep_parser(subparsers):
    """Add kit-sweep subcommand."""
    p = subparsers.add_parser(
        "kit-sweep",
        help="Run kit sweep (sub/special weapon analysis)",
        description="Analyze feature activation by sub weapon and special weapon",
    )
    _add_common_args(p)
    p.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top kits to show (default: 15)",
    )
    p.add_argument(
        "--min-examples",
        type=int,
        default=10,
        help="Minimum examples per kit (default: 10)",
    )
    p.add_argument(
        "--combinations",
        action="store_true",
        default=False,
        help="Also analyze sub+special combinations",
    )
    p.set_defaults(func=cmd_kit_sweep)


def _add_binary_parser(subparsers):
    """Add binary subcommand."""
    p = subparsers.add_parser(
        "binary",
        help="Run binary presence effect analysis",
        description="Analyze effect of binary abilities (comeback, stealth_jump, etc.)",
    )
    _add_common_args(p)
    p.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Comma-separated binary tokens to test (default: all binary abilities)",
    )
    p.add_argument(
        "--primary-family",
        type=str,
        default=None,
        help="Primary family for stratified analysis",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum samples per condition (default: 20)",
    )
    p.add_argument(
        "--high-percentile",
        type=int,
        default=90,
        help="Percentile threshold for high activation (default: 90)",
    )
    p.set_defaults(func=cmd_binary)


def _add_coverage_parser(subparsers):
    """Add coverage subcommand."""
    p = subparsers.add_parser(
        "coverage",
        help="Run core coverage analysis",
        description="Analyze token coverage in core vs tail activation regions",
    )
    _add_common_args(p)
    p.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="Comma-separated tokens to check coverage (default: auto-detect)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Coverage threshold for core concept (default: 0.30)",
    )
    p.add_argument(
        "--no-weapons",
        dest="include_weapons",
        action="store_false",
        default=True,
        help="Skip weapon coverage analysis",
    )
    p.set_defaults(func=cmd_coverage)


def _add_token_influence_parser(subparsers):
    """Add token-influence subcommand."""
    p = subparsers.add_parser(
        "token-influence",
        help="Run token influence sweep",
        description="Find enhancer and suppressor tokens based on activation",
    )
    _add_common_args(p)
    p.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples for token analysis (default: 50)",
    )
    p.add_argument(
        "--high-percentile",
        type=int,
        default=90,
        help="Percentile threshold for high activation (default: 90)",
    )
    p.add_argument(
        "--no-collapse",
        dest="collapse_families",
        action="store_false",
        default=True,
        help="Don't collapse tokens to family level",
    )
    p.set_defaults(func=cmd_token_influence)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for the runner CLI."""
    parser = argparse.ArgumentParser(
        description="Execute mechanistic interpretability experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (subcommand mode - recommended):
    # Family sweep
    python -m splatnlp.mechinterp.cli.runner_cli family-sweep \\
        --feature-id 6235 --family swim_speed_up

    # 2D heatmap
    python -m splatnlp.mechinterp.cli.runner_cli heatmap \\
        --feature-id 6235 --family-x quick_respawn --family-y ink_saver_sub

    # Weapon sweep
    python -m splatnlp.mechinterp.cli.runner_cli weapon-sweep \\
        --feature-id 6235 --top-k 20

Examples (JSON spec mode):
    python -m splatnlp.mechinterp.cli.runner_cli \\
        --spec-path /mnt/e/mechinterp_runs/specs/my_spec.json

    # List available experiment types
    python -m splatnlp.mechinterp.cli.runner_cli --list-types
        """,
    )

    # Legacy/utility flags (work without subcommand)
    parser.add_argument(
        "--spec-path",
        type=Path,
        help="Path to experiment spec JSON file (legacy mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write result JSON",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available experiment types and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate spec without running experiment (JSON mode only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="experiment subcommands",
        description="Run experiments directly without JSON specs",
    )

    _add_family_sweep_parser(subparsers)
    _add_heatmap_parser(subparsers)
    _add_weapon_sweep_parser(subparsers)
    _add_kit_sweep_parser(subparsers)
    _add_binary_parser(subparsers)
    _add_coverage_parser(subparsers)
    _add_token_influence_parser(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List types mode
    if args.list_types:
        print("Available experiment types:")
        print("-" * 40)
        runners = list_available_runners()
        for runner_name, types in sorted(runners.items()):
            print(f"\n{runner_name}:")
            for t in types:
                print(f"  - {t}")
        print("\n" + "-" * 40)
        print("Available subcommands:")
        print("  family-sweep, heatmap, weapon-sweep, kit-sweep,")
        print("  binary, coverage, token-influence")
        return 0

    # Subcommand mode - dispatch to handler
    if args.command:
        return args.func(args)

    # JSON spec mode
    if args.spec_path:
        return _run_from_spec(args)

    # No subcommand and no spec path - show help
    parser.print_help()
    return 0


def _run_from_spec(args) -> int:
    """Run experiment from JSON spec file (legacy mode)."""
    # Validate spec path
    if not args.spec_path.exists():
        logger.error(f"Spec file not found: {args.spec_path}")
        return 1

    # Load spec
    try:
        with open(args.spec_path) as f:
            spec_data = json.load(f)
        spec = ExperimentSpec.model_validate(spec_data)
        logger.info(
            f"Loaded spec: {spec.type.value} for feature {spec.feature_id}"
        )
    except Exception as e:
        logger.error(f"Failed to load spec: {e}")
        return 1

    # Dry run mode
    if args.dry_run:
        print("Spec validation successful:")
        print(f"  Type: {spec.type.value}")
        print(f"  Feature: {spec.feature_id}")
        print(f"  Model: {spec.model_type}")
        print(f"  Constraints: {spec.constraints}")
        return 0

    # Use shared execution helper
    return _execute_and_output(spec, args)


if __name__ == "__main__":
    sys.exit(main())
