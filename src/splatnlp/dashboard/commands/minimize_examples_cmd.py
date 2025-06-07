#!/usr/bin/env python3
"""
Select a minimal (or size-capped) set of examples that cover every (neuron, bin) pair
in a directory of neuron_*.json files.
"""

import argparse
import heapq
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Pair = Tuple[int, int]  # (neuron_id, bin_idx)
ExampleIndex = Dict[int, Set[Pair]]


def extract_id(ex: dict) -> int | None:
    """Return an example's unique ID, wherever the file puts it."""
    return (
        ex.get("example_id")  # ① flat field
        or ex.get("metadata", {}).get("example_id")  # ② your files
        or ex.get("index")  # ③ fallback some tools use
    )


def iter_neuron_objs(path: Path, bad_files: list[Path]) -> Iterator[dict]:
    """
    Yield every neuron-dict in *path*.  Append empty/invalid files to bad_files.
    """
    try:
        if path.stat().st_size == 0:
            bad_files.append(path)
            return
        with path.open() as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        bad_files.append(path)
        return

    if isinstance(data, dict) and "neuron_id" in data:
        yield data
    elif isinstance(data, list):
        yield from (d for d in data if isinstance(d, dict) and "neuron_id" in d)


def build_index(root: Path) -> ExampleIndex:
    """
    Load every *.json file in *root* and build a mapping:
        example_id  ->  {(neuron_id, bin_idx), ...}
    """
    ex2pairs: ExampleIndex = defaultdict(set)
    bad_files: list[Path] = []

    for path in root.glob("*.json"):
        for obj in iter_neuron_objs(path, bad_files):
            neuron_id = obj["neuron_id"]
            for bin_idx_str, bin_entry in obj["range_examples"].items():
                bin_idx = int(bin_idx_str)
                for ex in bin_entry["examples"]:
                    ex_id = extract_id(ex)
                    if ex_id is None:
                        continue
                    ex2pairs[ex_id].add((neuron_id, bin_idx))

    if bad_files:
        log.warning("Skipped %d empty/invalid JSON files:", len(bad_files))
        for p in bad_files:
            print("  •", p)

    return ex2pairs


def greedy_cover(
    ex2pairs: ExampleIndex, limit: int | None = None
) -> tuple[set[int], set[Pair]]:
    """
    Greedy set–cover.  Returns (chosen_example_ids, still_uncovered_pairs).
    If *limit* is given, stops once that many examples are selected.
    """
    uncovered: Set[Pair] = set().union(*ex2pairs.values())

    heap: List[Tuple[int, int]] = [
        (-len(pairs), ex) for ex, pairs in ex2pairs.items()
    ]
    heapq.heapify(heap)

    chosen: set[int] = set()
    while uncovered and heap and (limit is None or len(chosen) < limit):
        _, ex = heapq.heappop(heap)
        still_needed = ex2pairs[ex] & uncovered
        if not still_needed:  # stale heap entry
            continue
        chosen.add(ex)
        uncovered -= still_needed

    return chosen, uncovered


def minimize_examples_command(args: argparse.Namespace) -> None:
    """Command function for minimizing examples."""
    log.info("Indexing …")
    ex2pairs = build_index(args.root)
    log.info(f"  found {len(ex2pairs):,} unique examples")

    chosen, remaining = greedy_cover(ex2pairs, args.limit)
    log.info(
        f"Selected {len(chosen):,} examples; {len(remaining):,} pairs still uncovered"
    )

    ids = "\n".join(map(str, sorted(chosen)))
    if args.out:
        args.out.write_text(ids)
        log.info(f"IDs written to {args.out}")
    else:
        print(ids)


def setup_minimize_examples_parser(subparsers):
    """Set up the argument parser for the minimize-examples command."""
    p = subparsers.add_parser(
        "minimize-examples",
        help="Select a minimal set of examples covering all neuron/bin pairs.",
    )
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Directory containing neuron_*.json",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum #examples to return (optional)",
    )
    p.add_argument(
        "--out", type=Path, help="Write selected IDs here (one per line)"
    )
    p.set_defaults(func=minimize_examples_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pick a minimal set of examples covering all neuron/bin pairs."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Directory containing neuron_*.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum #examples to return (optional)",
    )
    parser.add_argument(
        "--out", type=Path, help="Write selected IDs here (one per line)"
    )
    args = parser.parse_args()
    minimize_examples_command(args)
