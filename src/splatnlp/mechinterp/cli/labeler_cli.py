"""CLI for feature labeling workflow.

Usage:
    # Get next feature to label
    poetry run python -m splatnlp.mechinterp.cli.labeler_cli next --model ultra

    # Set a label
    poetry run python -m splatnlp.mechinterp.cli.labeler_cli label \
        --feature-id 18712 --name "SCU Detector" --category tactical

    # Find similar features
    poetry run python -m splatnlp.mechinterp.cli.labeler_cli similar \
        --feature-id 18712 --top-k 5

    # Show labeling progress
    poetry run python -m splatnlp.mechinterp.cli.labeler_cli status --model ultra

    # Export labels to CSV
    poetry run python -m splatnlp.mechinterp.cli.labeler_cli export \
        --model ultra --output labels.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_next(args):
    """Get next feature to label."""
    from splatnlp.mechinterp.labeling import LabelingQueue, QueueBuilder

    queue = LabelingQueue.load(args.model)

    # If queue is empty, build one
    if len(queue) == 0 and not args.no_build:
        logger.info("Queue empty, building from activation counts...")
        builder = QueueBuilder(args.model)
        queue = builder.build_by_activation_count(top_k=50)

    entry = queue.get_next()
    if entry is None:
        print("Queue is empty. Use 'add' to add features or let it auto-build.")
        return

    print(f"\n## Next Feature: {entry.feature_id}")
    print(f"- Priority: {entry.priority:.3f}")
    print(f"- Reason: {entry.reason}")
    print(f"- Added: {entry.added_at.strftime('%Y-%m-%d %H:%M')}")

    # Show upcoming
    upcoming = queue.peek(5)
    if len(upcoming) > 1:
        print("\nUpcoming:")
        for i, e in enumerate(upcoming[1:], 2):
            print(f"  {i}. Feature {e.feature_id} (priority={e.priority:.3f})")


def cmd_label(args):
    """Set a label for a feature."""
    from splatnlp.mechinterp.labeling import LabelConsolidator, LabelingQueue

    consolidator = LabelConsolidator(args.model)
    queue = LabelingQueue.load(args.model)

    # Set the label
    label = consolidator.set_label(
        feature_id=args.feature_id,
        name=args.name,
        category=args.category,
        notes=args.notes or "",
        source=args.source,
    )

    # Mark complete in queue
    queue.mark_complete(args.feature_id, args.name)

    print(f"\nLabel saved for Feature {args.feature_id}:")
    print(f"- Name: {label.display_name}")
    print(f"- Category: {label.dashboard_category}")
    if label.dashboard_notes:
        print(f"- Notes: {label.dashboard_notes}")
    print(f"- Synced to: dashboard, research state, consolidated file")


def cmd_skip(args):
    """Skip a feature."""
    from splatnlp.mechinterp.labeling import LabelingQueue

    queue = LabelingQueue.load(args.model)
    feature_id = args.feature_id

    # If no feature specified, skip the next one
    if feature_id is None:
        entry = queue.get_next()
        if entry:
            feature_id = entry.feature_id
        else:
            print("Queue is empty.")
            return

    if queue.mark_skipped(feature_id, args.reason or ""):
        print(f"Skipped Feature {feature_id}")
        if args.reason:
            print(f"Reason: {args.reason}")
    else:
        print(f"Feature {feature_id} not found in queue")


def cmd_add(args):
    """Add feature(s) to the queue."""
    from splatnlp.mechinterp.labeling import LabelingQueue

    queue = LabelingQueue.load(args.model)

    feature_ids = [int(x) for x in args.feature_ids.split(",")]

    for fid in feature_ids:
        entry = queue.add(
            feature_id=fid,
            priority=args.priority,
            reason=args.reason or "manual add",
        )
        if entry:
            print(f"Added Feature {fid} (priority={args.priority})")


def cmd_similar(args):
    """Find features similar to a given feature."""
    from splatnlp.mechinterp.labeling import LabelConsolidator, SimilarFinder

    finder = SimilarFinder(args.model)
    consolidator = LabelConsolidator(args.model)
    consolidator.load_consolidated()

    print(f"\n## Features Similar to {args.feature_id}")

    similar = finder.find_by_top_tokens(args.feature_id, top_k=args.top_k)

    if not similar:
        print("No similar features found.")
        return

    for fid, sim in similar:
        label = consolidator.get_label(fid)
        label_str = ""
        if label and label.display_name:
            label_str = f" [{label.display_name}]"
        print(f"- Feature {fid}: similarity={sim:.3f}{label_str}")


def cmd_status(args):
    """Show labeling progress."""
    from splatnlp.mechinterp.labeling import LabelConsolidator, LabelingQueue

    consolidator = LabelConsolidator(args.model)
    consolidator.load_consolidated()
    queue = LabelingQueue.load(args.model)

    label_stats = consolidator.get_statistics()
    queue_stats = queue.get_statistics()

    print(f"\n## Labeling Status ({args.model})")
    print()
    print("### Labels")
    print(f"- Total labeled: {label_stats['total_labeled']}")
    print(f"- From dashboard: {label_stats['from_dashboard']}")
    print(f"- From research: {label_stats['from_research']}")
    print(f"- Merged: {label_stats['merged']}")
    print()
    print("### Categories")
    for cat, count in label_stats["by_category"].items():
        print(f"- {cat}: {count}")
    print()
    print("### Queue")
    print(f"- Pending: {queue_stats['pending']}")
    print(f"- Completed: {queue_stats['completed']}")
    print(f"- Skipped: {queue_stats['skipped']}")


def cmd_sync(args):
    """Sync labels from all sources."""
    from splatnlp.mechinterp.labeling import LabelConsolidator

    consolidator = LabelConsolidator(args.model)
    count = consolidator.sync_from_all_sources()
    print(f"Synced {count} labels from all sources")


def cmd_export(args):
    """Export labels to CSV."""
    from splatnlp.mechinterp.labeling import LabelConsolidator

    consolidator = LabelConsolidator(args.model)
    consolidator.load_consolidated()

    output_path = Path(args.output)
    consolidator.export_csv(output_path)
    print(f"Exported labels to {output_path}")


def cmd_build_queue(args):
    """Build a prioritized queue."""
    from splatnlp.mechinterp.labeling import QueueBuilder

    builder = QueueBuilder(args.model)

    if args.method == "activation_count":
        queue = builder.build_by_activation_count(
            top_k=args.top_k,
            exclude_labeled=not args.include_labeled,
        )
    elif args.method == "cluster":
        if not args.seed:
            print("--seed required for cluster method")
            return
        queue = builder.build_from_cluster(
            seed_feature=args.seed,
            top_k=args.top_k,
        )
    else:
        print(f"Unknown method: {args.method}")
        return

    print(f"Built queue with {len(queue)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Feature labeling workflow CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # next command
    next_parser = subparsers.add_parser("next", help="Get next feature to label")
    next_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type",
    )
    next_parser.add_argument(
        "--no-build",
        action="store_true",
        help="Don't auto-build queue if empty",
    )

    # label command
    label_parser = subparsers.add_parser("label", help="Set a feature label")
    label_parser.add_argument(
        "--feature-id", "-f",
        type=int,
        required=True,
        help="Feature ID to label",
    )
    label_parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        help="Label name",
    )
    label_parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["none", "mechanical", "tactical", "strategic"],
        default="none",
        help="Category",
    )
    label_parser.add_argument(
        "--notes",
        type=str,
        help="Additional notes",
    )
    label_parser.add_argument(
        "--source", "-s",
        type=str,
        default="dashboard",
        help="Label source (e.g., 'dashboard', 'claude code', 'research')",
    )
    label_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # skip command
    skip_parser = subparsers.add_parser("skip", help="Skip a feature")
    skip_parser.add_argument(
        "--feature-id", "-f",
        type=int,
        help="Feature ID to skip (defaults to next in queue)",
    )
    skip_parser.add_argument(
        "--reason", "-r",
        type=str,
        help="Reason for skipping",
    )
    skip_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # add command
    add_parser = subparsers.add_parser("add", help="Add feature(s) to queue")
    add_parser.add_argument(
        "feature_ids",
        type=str,
        help="Comma-separated feature IDs",
    )
    add_parser.add_argument(
        "--priority", "-p",
        type=float,
        default=0.5,
        help="Priority (0-1)",
    )
    add_parser.add_argument(
        "--reason", "-r",
        type=str,
        help="Reason for adding",
    )
    add_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar features")
    similar_parser.add_argument(
        "--feature-id", "-f",
        type=int,
        required=True,
        help="Seed feature ID",
    )
    similar_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of similar features to show",
    )
    similar_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Show labeling progress")
    status_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync labels from all sources")
    sync_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # export command
    export_parser = subparsers.add_parser("export", help="Export labels to CSV")
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        default="labels.csv",
        help="Output CSV path",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    # build-queue command
    build_parser = subparsers.add_parser("build-queue", help="Build prioritized queue")
    build_parser.add_argument(
        "--method",
        type=str,
        choices=["activation_count", "cluster"],
        default="activation_count",
        help="Prioritization method",
    )
    build_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of features to add",
    )
    build_parser.add_argument(
        "--seed",
        type=int,
        help="Seed feature for cluster method",
    )
    build_parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Include already-labeled features",
    )
    build_parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Dispatch to command handler
    commands = {
        "next": cmd_next,
        "label": cmd_label,
        "skip": cmd_skip,
        "add": cmd_add,
        "similar": cmd_similar,
        "status": cmd_status,
        "sync": cmd_sync,
        "export": cmd_export,
        "build-queue": cmd_build_queue,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
