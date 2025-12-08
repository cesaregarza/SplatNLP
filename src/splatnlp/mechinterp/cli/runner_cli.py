"""CLI entry point for the mechinterp-runner skill.

This module provides the command-line interface for executing
experiment specifications.

Usage:
    poetry run python -m splatnlp.mechinterp.cli.runner_cli \\
        --spec-path /mnt/e/mechinterp_runs/specs/20250607__f42__family-1d-sweep.json \\
        --output-dir /mnt/e/mechinterp_runs/results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from splatnlp.mechinterp.experiments.base import (
    get_runner_for_type,
    list_available_runners,
)
from splatnlp.mechinterp.schemas.experiment_specs import ExperimentSpec
from splatnlp.mechinterp.skill_helpers.context_loader import load_context
from splatnlp.mechinterp.state.io import RESULTS_DIR, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for the runner CLI."""
    parser = argparse.ArgumentParser(
        description="Execute mechanistic interpretability experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run a single experiment spec
    python -m splatnlp.mechinterp.cli.runner_cli \\
        --spec-path /mnt/e/mechinterp_runs/specs/my_spec.json

    # Run with custom output directory
    python -m splatnlp.mechinterp.cli.runner_cli \\
        --spec-path my_spec.json \\
        --output-dir ./results/

    # List available experiment types
    python -m splatnlp.mechinterp.cli.runner_cli --list-types
        """,
    )

    parser.add_argument(
        "--spec-path",
        type=Path,
        help="Path to experiment spec JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write result JSON (default: /mnt/e/mechinterp_runs/results/)",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available experiment types and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate spec without running experiment",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

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
        return 0

    # Normal run mode - require spec path
    if not args.spec_path:
        parser.error("--spec-path is required (unless using --list-types)")

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
        print(f"Spec validation successful:")
        print(f"  Type: {spec.type.value}")
        print(f"  Feature: {spec.feature_id}")
        print(f"  Model: {spec.model_type}")
        print(f"  Constraints: {spec.constraints}")
        return 0

    # Ensure output directory exists
    ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Write result
    output_path = args.output_dir / result.to_filename()
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

    if result.success:
        print(f"\nResult saved to: {output_path}")
        return 0
    else:
        print(f"\nExperiment FAILED: {result.error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
