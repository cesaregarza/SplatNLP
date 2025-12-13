"""Frequent itemset mining experiment runner.

This module implements frequent itemset analysis to find co-occurring
token patterns in high-activation examples.
"""

import logging
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from splatnlp.mechinterp.experiments.base import (
    ExperimentRunner,
    register_runner,
)
from splatnlp.mechinterp.schemas.experiment_results import ExperimentResult
from splatnlp.mechinterp.schemas.experiment_specs import (
    ExperimentSpec,
    ExperimentType,
)
from splatnlp.mechinterp.schemas.glossary import parse_token
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


@register_runner
class FrequentItemsetRunner(ExperimentRunner):
    """Runner for frequent itemset mining experiments.

    Finds co-occurring token patterns that are enriched in
    high-activation examples compared to baseline.
    """

    name = "frequent_itemsets"
    handles_types = [ExperimentType.FREQUENT_ITEMSETS]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute frequent itemset analysis."""
        vars_dict = spec.variables
        min_support = vars_dict.get("min_support", 0.05)
        max_size = vars_dict.get("max_size", 4)
        collapse_families = vars_dict.get("collapse_families", False)
        condition_on = vars_dict.get("condition_on")
        high_pct = vars_dict.get("high_activation_pct", 10.0)

        # Get activation data
        try:
            data = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            data = ctx.db.get_feature_activations(spec.feature_id, limit=10000)

        # Handle both DataFrame and list returns
        import polars as pl

        if isinstance(data, pl.DataFrame):
            if data.is_empty():
                raise ValueError(f"No data for feature {spec.feature_id}")
            # Convert DataFrame to list of dicts for compatibility
            data = data.to_dicts()
        elif not data:
            raise ValueError(f"No data for feature {spec.feature_id}")

        # Sort by activation and split into high/baseline
        data = sorted(data, key=lambda x: x.get("activation", 0), reverse=True)
        n_high = max(1, int(len(data) * high_pct / 100))
        high_data = data[:n_high]
        baseline_data = data[n_high:]

        logger.info(
            f"Analyzing {n_high} high-activation examples "
            f"vs {len(baseline_data)} baseline"
        )

        # Extract token sets
        high_sets = self._extract_token_sets(
            high_data, ctx, collapse_families, condition_on
        )
        baseline_sets = self._extract_token_sets(
            baseline_data, ctx, collapse_families, condition_on
        )

        # Mine frequent itemsets
        itemsets = []
        for size in range(2, max_size + 1):
            size_itemsets = self._mine_itemsets(
                high_sets, baseline_sets, size, min_support
            )
            itemsets.extend(size_itemsets)

        # Sort by lift
        itemsets.sort(key=lambda x: x["lift"], reverse=True)

        # Add to result
        result.add_table(
            "itemsets",
            itemsets[:100],  # Top 100
            columns=[
                "tokens",
                "size",
                "support_high",
                "support_baseline",
                "lift",
                "count",
            ],
            description=f"Frequent itemsets (min_support={min_support})",
        )

        # Aggregates
        if itemsets:
            result.aggregates.custom["n_itemsets"] = len(itemsets)
            result.aggregates.custom["max_lift"] = itemsets[0]["lift"]
            result.aggregates.custom["avg_lift"] = sum(
                i["lift"] for i in itemsets
            ) / len(itemsets)

        result.aggregates.n_samples = len(data)
        result.diagnostics.n_contexts_tested = len(high_data)

    def _extract_token_sets(
        self,
        data: list[dict],
        ctx: MechInterpContext,
        collapse_families: bool,
        condition_on: str | None,
    ) -> list[frozenset[str]]:
        """Extract token sets from examples."""
        sets = []
        for row in data:
            # Support both column names for compatibility
            tokens = row.get("tokens") or row.get("ability_input_tokens", [])
            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            # Filter out empty and special tokens
            tokens = [t for t in tokens if t and not t.startswith("<")]

            # Apply conditioning
            if condition_on:
                if condition_on not in tokens:
                    continue  # Skip examples without condition token
                tokens = [t for t in tokens if t != condition_on]

            # Collapse families if requested
            if collapse_families:
                tokens = [parse_token(t)[0] for t in tokens]
                tokens = list(set(tokens))  # Deduplicate

            if tokens:
                sets.append(frozenset(tokens))

        return sets

    def _mine_itemsets(
        self,
        high_sets: list[frozenset],
        baseline_sets: list[frozenset],
        size: int,
        min_support: float,
    ) -> list[dict]:
        """Mine frequent itemsets of given size."""
        if not high_sets:
            return []

        # Count itemsets in high activation
        high_counter: Counter = Counter()
        for token_set in high_sets:
            if len(token_set) >= size:
                for combo in combinations(sorted(token_set), size):
                    high_counter[combo] += 1

        # Filter by support
        n_high = len(high_sets)
        min_count = int(n_high * min_support)

        # Count in baseline for comparison
        n_baseline = len(baseline_sets)
        baseline_counter: Counter = Counter()
        for token_set in baseline_sets:
            if len(token_set) >= size:
                for combo in combinations(sorted(token_set), size):
                    baseline_counter[combo] += 1

        # Compute lift
        results = []
        for itemset, count in high_counter.items():
            if count < min_count:
                continue

            support_high = count / n_high
            baseline_count = baseline_counter.get(itemset, 0)
            support_baseline = (
                baseline_count / n_baseline if n_baseline > 0 else 0
            )

            # Lift = observed / expected
            lift = (
                support_high / support_baseline
                if support_baseline > 0
                else float("inf")
            )

            results.append(
                {
                    "tokens": " + ".join(itemset),
                    "size": size,
                    "support_high": round(support_high, 4),
                    "support_baseline": round(support_baseline, 4),
                    "lift": round(lift, 2) if lift != float("inf") else 999.0,
                    "count": count,
                }
            )

        return results
