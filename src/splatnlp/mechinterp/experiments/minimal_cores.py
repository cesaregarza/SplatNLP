"""Minimal activating core experiment runner.

This module implements analysis to find the minimal set of tokens
that maintain feature activation.
"""

import logging
from collections import Counter
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
from splatnlp.mechinterp.schemas.glossary import parse_token
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


@register_runner
class MinimalCoreRunner(ExperimentRunner):
    """Runner for minimal activating core analysis.

    Finds the smallest subset of tokens that maintains a threshold
    fraction of the original activation.
    """

    name = "minimal_cores"
    handles_types = [ExperimentType.MINIMAL_CORES]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute minimal core analysis.

        Note: This is a simplified version that analyzes token frequency
        in high-activation examples. Full minimal core computation would
        require model access to test token removal effects.
        """
        vars_dict = spec.variables
        clamp_families = vars_dict.get("clamp_families", [])
        exclude_families = vars_dict.get("exclude_families", [])
        max_examples = vars_dict.get("max_examples", 100)

        # Get high-activation examples
        try:
            data = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            data = ctx.db.get_feature_activations(
                spec.feature_id, limit=max_examples
            )

        # Handle both DataFrame and list returns
        if isinstance(data, pl.DataFrame):
            if data.is_empty():
                raise ValueError(f"No data for feature {spec.feature_id}")
            data = data.to_dicts()
        elif not data:
            raise ValueError(f"No data for feature {spec.feature_id}")

        # Sort by activation and take top examples
        data = sorted(data, key=lambda x: x.get("activation", 0), reverse=True)
        data = data[:max_examples]

        logger.info(f"Analyzing {len(data)} high-activation examples")

        # Count token frequencies in high-activation examples
        token_counts: Counter = Counter()
        family_counts: Counter = Counter()
        n_examples = 0

        for row in data:
            tokens = row.get("tokens", [])
            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            # Filter special tokens
            tokens = [t for t in tokens if t and not t.startswith("<")]

            # Apply family filters
            filtered_tokens = []
            for t in tokens:
                family, _ = parse_token(t)
                if family in exclude_families:
                    continue
                filtered_tokens.append(t)
                family_counts[family] += 1

            token_counts.update(filtered_tokens)
            n_examples += 1

        # Find "core" tokens (appear in >50% of high-activation examples)
        core_threshold = 0.5
        core_tokens = [
            (tok, count, count / n_examples)
            for tok, count in token_counts.most_common()
            if count / n_examples >= core_threshold
        ]

        # Build result tables
        result.add_table(
            "token_frequency",
            [
                {"token": tok, "count": count, "frequency": round(freq, 3)}
                for tok, count, freq in core_tokens[:50]
            ],
            description="Most frequent tokens in high-activation examples",
        )

        result.add_table(
            "family_frequency",
            [
                {
                    "family": fam,
                    "count": count,
                    "frequency": round(count / n_examples, 3),
                }
                for fam, count in family_counts.most_common(20)
            ],
            description="Most frequent ability families",
        )

        # Aggregates
        result.aggregates.n_samples = n_examples
        result.aggregates.custom["n_core_tokens"] = len(core_tokens)
        if core_tokens:
            result.aggregates.custom["top_token"] = core_tokens[0][0]
            result.aggregates.custom["top_token_freq"] = core_tokens[0][2]

        result.diagnostics.n_contexts_tested = n_examples
        result.diagnostics.warnings.append(
            "Simplified analysis: token frequency in high-activation examples. "
            "Full minimal core computation requires model access."
        )
