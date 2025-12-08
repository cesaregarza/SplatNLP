"""Interaction analysis experiment runners.

This module implements pairwise and conditional interaction analysis
to understand synergy and redundancy between tokens.
"""

import logging
from collections import defaultdict
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
class PairwiseInteractionRunner(ExperimentRunner):
    """Runner for pairwise interaction analysis.

    Estimates synergy/redundancy between token pairs by analyzing
    co-occurrence patterns in activation data.
    """

    name = "pairwise_interactions"
    handles_types = [ExperimentType.PAIRWISE_INTERACTIONS]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute pairwise interaction analysis.

        Note: This is an observational analysis based on co-occurrence
        and activation correlation. True causal interactions require
        model access for interventional experiments.
        """
        vars_dict = spec.variables
        candidate_tokens = vars_dict.get("candidate_tokens")
        n_candidates = vars_dict.get("n_candidates", 20)
        family_mode = vars_dict.get("family_mode", False)

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

        logger.info(f"Analyzing {len(data)} examples")

        # If no candidates specified, use most frequent tokens
        if not candidate_tokens:
            candidate_tokens = self._get_top_tokens(
                data, ctx, n_candidates, family_mode
            )

        logger.info(f"Analyzing {len(candidate_tokens)} candidate tokens")

        # Compute pairwise statistics
        pair_stats = self._compute_pair_stats(
            data, ctx, candidate_tokens, family_mode
        )

        # Sort by interaction strength (deviation from independence)
        pair_stats.sort(key=lambda x: abs(x["interaction"]), reverse=True)

        # Separate synergies and redundancies
        synergies = [p for p in pair_stats if p["interaction"] > 0]
        redundancies = [p for p in pair_stats if p["interaction"] < 0]

        result.add_table(
            "synergies",
            synergies[:30],
            columns=[
                "token_a",
                "token_b",
                "interaction",
                "cooccurrence_rate",
                "n_both",
            ],
            description="Token pairs with synergistic effects (positive interaction)",
        )

        result.add_table(
            "redundancies",
            redundancies[:30],
            columns=[
                "token_a",
                "token_b",
                "interaction",
                "cooccurrence_rate",
                "n_both",
            ],
            description="Token pairs with redundant effects (negative interaction)",
        )

        # Aggregates
        if pair_stats:
            result.aggregates.custom["n_pairs_analyzed"] = len(pair_stats)
            result.aggregates.custom["n_synergies"] = len(synergies)
            result.aggregates.custom["n_redundancies"] = len(redundancies)
            result.aggregates.custom["max_synergy"] = (
                synergies[0]["interaction"] if synergies else 0
            )
            result.aggregates.custom["max_redundancy"] = (
                redundancies[0]["interaction"] if redundancies else 0
            )

        result.aggregates.n_samples = len(data)
        result.diagnostics.n_contexts_tested = len(data)

    def _get_top_tokens(
        self,
        data: list[dict],
        ctx: MechInterpContext,
        n: int,
        family_mode: bool,
    ) -> list[str]:
        """Get top N most frequent tokens/families."""
        from collections import Counter

        counter: Counter = Counter()

        for row in data:
            # Support both column names for compatibility
            tokens = row.get("tokens") or row.get("ability_input_tokens", [])
            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            for t in tokens:
                if t and not t.startswith("<"):
                    if family_mode:
                        t = parse_token(t)[0]
                    counter[t] += 1

        return [tok for tok, _ in counter.most_common(n)]

    def _compute_pair_stats(
        self,
        data: list[dict],
        ctx: MechInterpContext,
        candidates: list[str],
        family_mode: bool,
    ) -> list[dict]:
        """Compute statistics for all candidate pairs."""
        # Count individual and pair occurrences
        single_counts: dict[str, int] = defaultdict(int)
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        activation_sums: dict[tuple[str, str], float] = defaultdict(float)
        n_total = 0

        for row in data:
            # Support both column names for compatibility
            tokens = row.get("tokens") or row.get("ability_input_tokens", [])
            activation = row.get("activation", 0)

            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            # Filter and optionally collapse to families
            filtered = []
            for t in tokens:
                if t and not t.startswith("<"):
                    if family_mode:
                        t = parse_token(t)[0]
                    if t in candidates:
                        filtered.append(t)

            filtered = list(set(filtered))  # Deduplicate
            n_total += 1

            for t in filtered:
                single_counts[t] += 1

            for a, b in combinations(sorted(filtered), 2):
                pair = (a, b)
                pair_counts[pair] += 1
                activation_sums[pair] += activation

        # Compute interaction scores
        results = []
        for (a, b), n_both in pair_counts.items():
            if n_both < 5:  # Skip rare pairs
                continue

            # Expected co-occurrence under independence
            p_a = single_counts[a] / n_total
            p_b = single_counts[b] / n_total
            expected = p_a * p_b * n_total

            # Interaction: observed - expected (normalized)
            observed = n_both
            interaction = (observed - expected) / max(expected, 1)

            # Mean activation when both present
            mean_act = activation_sums[(a, b)] / n_both

            results.append(
                {
                    "token_a": a,
                    "token_b": b,
                    "interaction": round(interaction, 3),
                    "cooccurrence_rate": round(n_both / n_total, 4),
                    "n_both": n_both,
                    "mean_activation": round(mean_act, 4),
                }
            )

        return results


@register_runner
class ConditionalInteractionRunner(ExperimentRunner):
    """Runner for conditional interaction analysis.

    Analyzes how a third token modulates pairwise interactions.
    """

    name = "conditional_interactions"
    handles_types = [ExperimentType.CONDITIONAL_INTERACTIONS]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute conditional interaction analysis."""
        vars_dict = spec.variables
        modulator = vars_dict.get("modulator_token")

        if not modulator:
            raise ValueError("Missing 'modulator_token' in variables")

        # First run basic pairwise analysis
        base_runner = PairwiseInteractionRunner()
        base_runner._run_experiment(spec, ctx, result)

        # Add note about conditional analysis
        result.diagnostics.warnings.append(
            f"Conditional analysis on '{modulator}' requires model access. "
            "Currently showing base pairwise analysis."
        )
