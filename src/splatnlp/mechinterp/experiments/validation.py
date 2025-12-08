"""Validation experiment runners.

This module implements credibility checks including split-half
correlation and shuffle null tests.
"""

import logging
import random
import statistics
from typing import Any

import polars as pl

from splatnlp.mechinterp.experiments.base import (
    ExperimentRunner,
    register_runner,
)
from splatnlp.mechinterp.schemas.experiment_results import ExperimentResult
from splatnlp.mechinterp.schemas.experiment_specs import (
    ExperimentSpec,
    ExperimentType,
)
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


@register_runner
class SplitHalfRunner(ExperimentRunner):
    """Runner for split-half validation.

    Tests stability of analysis results by computing correlation
    across random splits of the data.
    """

    name = "split_half"
    handles_types = [ExperimentType.SPLIT_HALF]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute split-half validation."""
        vars_dict = spec.variables
        n_splits = vars_dict.get("n_splits", 10)
        metric = vars_dict.get("metric", "token_frequency_correlation")

        # Get activation data
        try:
            data = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            data = ctx.db.get_feature_activations(spec.feature_id, limit=10000)

        # Handle both DataFrame and list returns
        if isinstance(data, pl.DataFrame):
            if data.is_empty():
                raise ValueError(f"No data for feature {spec.feature_id}")
            data = data.to_dicts()
        elif not data:
            raise ValueError(f"No data for feature {spec.feature_id}")

        logger.info(
            f"Running {n_splits} split-half validations on {len(data)} examples"
        )

        # Run multiple splits
        correlations = []
        for i in range(n_splits):
            random.seed(42 + i)
            shuffled = data.copy()
            random.shuffle(shuffled)

            mid = len(shuffled) // 2
            half_a = shuffled[:mid]
            half_b = shuffled[mid:]

            corr = self._compute_split_correlation(half_a, half_b, ctx, metric)
            correlations.append(corr)

        # Compute statistics
        mean_corr = statistics.mean(correlations)
        std_corr = (
            statistics.stdev(correlations) if len(correlations) > 1 else 0
        )

        # Results table
        result.add_table(
            "split_correlations",
            [
                {"split": i + 1, "correlation": round(c, 4)}
                for i, c in enumerate(correlations)
            ],
            description=f"Split-half correlations for {metric}",
        )

        # Aggregates
        result.aggregates.custom["mean_correlation"] = round(mean_corr, 4)
        result.aggregates.custom["std_correlation"] = round(std_corr, 4)
        result.aggregates.custom["min_correlation"] = round(
            min(correlations), 4
        )
        result.aggregates.custom["max_correlation"] = round(
            max(correlations), 4
        )

        # Pass/fail threshold
        stability_threshold = 0.7
        passed = mean_corr >= stability_threshold
        result.aggregates.custom["stability_passed"] = 1 if passed else 0

        result.aggregates.n_samples = len(data)
        result.diagnostics.n_contexts_tested = n_splits

        if not passed:
            result.diagnostics.warnings.append(
                f"Split-half correlation ({mean_corr:.2f}) below threshold ({stability_threshold})"
            )

    def _compute_split_correlation(
        self,
        half_a: list[dict],
        half_b: list[dict],
        ctx: MechInterpContext,
        metric: str,
    ) -> float:
        """Compute correlation between two data halves."""
        # Compute token frequency in each half
        freq_a = self._compute_token_frequencies(half_a, ctx)
        freq_b = self._compute_token_frequencies(half_b, ctx)

        # Get common tokens
        common = set(freq_a.keys()) & set(freq_b.keys())
        if len(common) < 5:
            return 0.0

        # Compute Pearson correlation
        values_a = [freq_a[t] for t in common]
        values_b = [freq_b[t] for t in common]

        return self._pearson_correlation(values_a, values_b)

    def _compute_token_frequencies(
        self,
        data: list[dict],
        ctx: MechInterpContext,
    ) -> dict[str, float]:
        """Compute token frequency weighted by activation."""
        from collections import Counter

        weighted_counts: Counter = Counter()
        total_weight = 0

        for row in data:
            tokens = row.get("tokens", [])
            activation = row.get("activation", 1.0)

            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            for t in tokens:
                if t and not t.startswith("<"):
                    weighted_counts[t] += activation
                    total_weight += activation

        if total_weight == 0:
            return {}

        return {t: c / total_weight for t, c in weighted_counts.items()}

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)


@register_runner
class ShuffleNullRunner(ExperimentRunner):
    """Runner for shuffle null validation.

    Creates a null distribution by shuffling activations to test
    significance of observed patterns.
    """

    name = "shuffle_null"
    handles_types = [ExperimentType.SHUFFLE_NULL]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute shuffle null validation."""
        vars_dict = spec.variables
        n_shuffles = vars_dict.get("n_shuffles", 100)

        # Get activation data
        try:
            data = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            data = ctx.db.get_feature_activations(spec.feature_id, limit=10000)

        # Handle both DataFrame and list returns
        if isinstance(data, pl.DataFrame):
            if data.is_empty():
                raise ValueError(f"No data for feature {spec.feature_id}")
            data = data.to_dicts()
        elif not data:
            raise ValueError(f"No data for feature {spec.feature_id}")

        logger.info(
            f"Running {n_shuffles} shuffle null tests on {len(data)} examples"
        )

        # Compute observed statistic (top token frequency in high-activation)
        data_sorted = sorted(
            data, key=lambda x: x.get("activation", 0), reverse=True
        )
        n_high = max(1, len(data) // 10)  # Top 10%

        observed_stat = self._compute_top_token_concentration(
            data_sorted[:n_high], ctx
        )

        # Generate null distribution
        null_stats = []
        activations = [row.get("activation", 0) for row in data]

        for i in range(n_shuffles):
            random.seed(42 + i)
            shuffled_acts = activations.copy()
            random.shuffle(shuffled_acts)

            # Reassign activations
            shuffled_data = [
                {**row, "activation": act}
                for row, act in zip(data, shuffled_acts)
            ]
            shuffled_sorted = sorted(
                shuffled_data,
                key=lambda x: x.get("activation", 0),
                reverse=True,
            )

            null_stat = self._compute_top_token_concentration(
                shuffled_sorted[:n_high], ctx
            )
            null_stats.append(null_stat)

        # Compute p-value
        n_exceed = sum(1 for ns in null_stats if ns >= observed_stat)
        p_value = n_exceed / n_shuffles

        # Results
        result.add_table(
            "null_distribution",
            [
                {
                    "observed": round(observed_stat, 4),
                    "null_mean": round(statistics.mean(null_stats), 4),
                    "null_std": round(
                        (
                            statistics.stdev(null_stats)
                            if len(null_stats) > 1
                            else 0
                        ),
                        4,
                    ),
                    "p_value": round(p_value, 4),
                }
            ],
            description="Shuffle null test results",
        )

        # Aggregates
        result.aggregates.custom["observed_stat"] = round(observed_stat, 4)
        result.aggregates.custom["null_mean"] = round(
            statistics.mean(null_stats), 4
        )
        result.aggregates.custom["p_value"] = round(p_value, 4)
        result.aggregates.custom["significant"] = 1 if p_value < 0.05 else 0

        result.aggregates.n_samples = len(data)
        result.diagnostics.n_contexts_tested = n_shuffles

        if p_value >= 0.05:
            result.diagnostics.warnings.append(
                f"Pattern not significant (p={p_value:.3f} >= 0.05)"
            )

    def _compute_top_token_concentration(
        self,
        data: list[dict],
        ctx: MechInterpContext,
    ) -> float:
        """Compute concentration of top token in examples."""
        from collections import Counter

        counter: Counter = Counter()
        for row in data:
            tokens = row.get("tokens", [])
            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            for t in tokens:
                if t and not t.startswith("<"):
                    counter[t] += 1

        if not counter:
            return 0.0

        # Return fraction held by top token
        total = sum(counter.values())
        top_count = counter.most_common(1)[0][1]
        return top_count / total
