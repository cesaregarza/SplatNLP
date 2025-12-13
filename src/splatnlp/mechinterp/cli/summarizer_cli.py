"""CLI entry point for the mechinterp-summarizer skill.

This module provides the command-line interface for converting
experiment results into research notes and updating state.

Usage:
    poetry run python -m splatnlp.mechinterp.cli.summarizer_cli \\
        --result-path /mnt/e/mechinterp_runs/results/20250607__result.json \\
        --feature-id 18712 \\
        --model-type ultra
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from splatnlp.mechinterp.schemas.experiment_results import ExperimentResult
from splatnlp.mechinterp.schemas.research_state import EvidenceStrength
from splatnlp.mechinterp.state import ResearchStateManager
from splatnlp.mechinterp.state.io import NOTES_DIR, RESULTS_DIR, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_note_from_result(result: ExperimentResult) -> str:
    """Generate a Markdown research note from experiment result."""
    lines = [
        f"## {result.experiment_type.replace('_', ' ').title()}",
        f"*Experiment: {result.spec_id}*",
        f"*Duration: {result.duration_seconds:.1f}s*",
        "",
    ]

    if not result.success:
        lines.extend(
            [
                "### Status: FAILED",
                f"Error: {result.error_message}",
                "",
            ]
        )
        return "\n".join(lines)

    # Key findings
    lines.append("### Key Findings")

    if result.aggregates.mean_delta is not None:
        lines.append(
            f"- Mean activation delta: **{result.aggregates.mean_delta:.4f}**"
        )
    if result.aggregates.max_delta is not None:
        lines.append(f"- Max delta: {result.aggregates.max_delta:.4f}")
    if result.aggregates.n_samples > 0:
        lines.append(f"- Samples analyzed: {result.aggregates.n_samples}")

    for key, val in result.aggregates.custom.items():
        if isinstance(val, float):
            lines.append(f"- {key.replace('_', ' ').title()}: {val:.4f}")
        else:
            lines.append(f"- {key.replace('_', ' ').title()}: {val}")

    lines.append("")

    # Tables (abbreviated)
    for table_name, table in result.tables.items():
        lines.append(f"### {table.name}")
        if table.description:
            lines.append(f"*{table.description}*")
        lines.append("")
        lines.append(table.to_markdown(max_rows=10))
        lines.append("")

    # Diagnostics
    if result.diagnostics.warnings:
        lines.append("### Warnings")
        for w in result.diagnostics.warnings:
            lines.append(f"- {w}")
        lines.append("")

    if result.diagnostics.relu_floor_detected:
        lines.append(
            f"**ReLU floor detected** ({result.diagnostics.relu_floor_rate:.1%} of contexts)"
        )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main entry point for the summarizer CLI."""
    parser = argparse.ArgumentParser(
        description="Convert experiment results to research notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
        help="Path to experiment result JSON file",
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        help="Feature ID (auto-detected from result if not provided)",
    )
    parser.add_argument(
        "--model-type",
        choices=["full", "ultra"],
        default="ultra",
        help="Model type (default: ultra)",
    )
    parser.add_argument(
        "--update-state",
        action="store_true",
        help="Update research state with evidence",
    )
    parser.add_argument(
        "--supports",
        nargs="*",
        help="Hypothesis IDs this evidence supports (e.g., h001 h002)",
    )
    parser.add_argument(
        "--refutes",
        nargs="*",
        help="Hypothesis IDs this evidence refutes",
    )
    parser.add_argument(
        "--strength",
        choices=["strong", "moderate", "weak"],
        default="moderate",
        help="Evidence strength (default: moderate)",
    )
    parser.add_argument(
        "--output-only",
        action="store_true",
        help="Only print note to stdout, don't write files",
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

    # Load result
    if not args.result_path.exists():
        logger.error(f"Result file not found: {args.result_path}")
        return 1

    try:
        with open(args.result_path) as f:
            result_data = json.load(f)
        result = ExperimentResult.model_validate(result_data)
        logger.info(
            f"Loaded result: {result.experiment_type} for feature {result.feature_id}"
        )
    except Exception as e:
        logger.error(f"Failed to load result: {e}")
        return 1

    # Determine feature ID
    feature_id = args.feature_id or result.feature_id

    # Generate note
    note = generate_note_from_result(result)

    if args.output_only:
        print(note)
        return 0

    # Ensure directories exist
    ensure_dirs()

    # Write/append note to file
    notes_path = NOTES_DIR / f"feature_{feature_id}_{args.model_type}.md"

    if notes_path.exists():
        existing = notes_path.read_text()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_content = f"{existing}\n\n---\n*Added: {timestamp}*\n\n{note}"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_content = f"# Feature {feature_id} Research Notes\n*Model: {args.model_type}*\n*Created: {timestamp}*\n\n{note}"

    notes_path.write_text(new_content)
    logger.info(f"Notes written to: {notes_path}")

    # Update research state if requested
    if args.update_state:
        manager = ResearchStateManager(
            feature_id=feature_id,
            model_type=args.model_type,
        )

        strength_map = {
            "strong": EvidenceStrength.STRONG,
            "moderate": EvidenceStrength.MODERATE,
            "weak": EvidenceStrength.WEAK,
        }

        evidence = manager.add_evidence(
            experiment_id=result.spec_id,
            result_path=str(args.result_path),
            summary=result.get_summary()[:200],
            strength=strength_map[args.strength],
            supports=args.supports or [],
            refutes=args.refutes or [],
            key_metrics={
                k: v
                for k, v in result.aggregates.custom.items()
                if isinstance(v, (int, float))
            },
        )
        logger.info(f"Added evidence {evidence.id} to state")

    # Print summary
    print("\n" + "=" * 60)
    print(note)
    print("=" * 60)
    print(f"\nNotes saved to: {notes_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
