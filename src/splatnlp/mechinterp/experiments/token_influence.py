"""Token influence sweep experiment runner.

This module implements the token_influence_sweep experiment which analyzes
how each token family affects feature activation, distinguishing between
enhancers (tokens that promote high activation) and suppressors (tokens
that are excluded from high-activation examples).
"""

import logging
import math
import statistics
from collections import defaultdict
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
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


def _token_base(token: str) -> str:
    """Strip AP suffixes like _57 to get the ability family name."""
    if not token:
        return ""
    parts = token.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    return token


@register_runner
class TokenInfluenceSweepRunner(ExperimentRunner):
    """Runner for token influence sweep experiments.

    Analyzes how each token (ability family) influences feature activation
    by computing:
    - high_rate_ratio: How often the token appears in high vs low activation examples
    - delta: Mean activation difference when token is present vs absent
    - effect_size: Cohen's d for the delta

    Tokens are classified as:
    - Enhancers: high_rate_ratio > threshold (appear more in high-activation examples)
    - Suppressors: high_rate_ratio < threshold (excluded from high-activation examples)
    """

    name = "token_influence_sweep"
    handles_types = [ExperimentType.TOKEN_INFLUENCE_SWEEP]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute token influence sweep analysis."""
        # Parse variables
        vars_dict = spec.variables
        min_samples = vars_dict.get("min_samples", 50)
        high_percentile = vars_dict.get("high_percentile", 0.995)
        collapse_families = vars_dict.get("collapse_families", True)
        suppressor_threshold = vars_dict.get("suppressor_threshold", 0.8)
        enhancer_threshold = vars_dict.get("enhancer_threshold", 1.2)

        logger.info(
            f"Running token influence sweep (high_percentile={high_percentile}, "
            f"min_samples={min_samples})"
        )

        # Get activation data
        logger.info("Loading activation data...")
        try:
            df = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            df = ctx.db.get_feature_activations(spec.feature_id, limit=50000)

        if df is None or len(df) == 0:
            raise ValueError(
                f"No activation data for feature {spec.feature_id}"
            )

        logger.info(f"Loaded {len(df)} examples")

        # Compute per-token statistics
        token_present_activations: dict[str, list[float]] = defaultdict(list)
        all_activations: list[float] = []

        for row in df.iter_rows(named=True):
            token_ids = row.get("ability_input_tokens", [])
            activation = row.get("activation", 0)
            all_activations.append(activation)

            bases_seen = set()
            for tid in token_ids or []:
                tname = ctx.inv_vocab.get(tid, "")
                if collapse_families:
                    base = _token_base(tname)
                else:
                    base = tname
                if base:
                    bases_seen.add(base)

            for base in bases_seen:
                token_present_activations[base].append(activation)

        n_total = len(all_activations)
        if n_total == 0:
            raise ValueError("No activations found")

        # Compute high activation threshold
        sorted_acts = sorted(all_activations, reverse=True)
        high_idx = max(1, int(n_total * (1 - high_percentile)))
        high_threshold = sorted_acts[min(high_idx, len(sorted_acts) - 1)]

        n_high = sum(1 for a in all_activations if a >= high_threshold)
        n_low = n_total - n_high

        logger.info(
            f"High activation threshold: {high_threshold:.4f} "
            f"(n_high={n_high}, n_low={n_low})"
        )

        sum_all = sum(all_activations)

        # Compute influence for each token
        token_stats: list[dict[str, Any]] = []
        suppressors: list[dict[str, Any]] = []
        enhancers: list[dict[str, Any]] = []

        for token, present_acts in token_present_activations.items():
            n_present = len(present_acts)
            n_absent = n_total - n_present

            if n_present < min_samples or n_absent < min_samples:
                continue

            # Count high/low activation examples containing this token
            n_high_with_token = sum(
                1 for a in present_acts if a >= high_threshold
            )
            n_low_with_token = n_present - n_high_with_token

            # Compute rates
            high_rate = n_high_with_token / n_high if n_high > 0 else 0
            low_rate = n_low_with_token / n_low if n_low > 0 else 0

            # Compute high_rate_ratio
            if low_rate > 0.001:
                high_rate_ratio = high_rate / low_rate
            elif high_rate > 0:
                high_rate_ratio = 10.0
            else:
                high_rate_ratio = 1.0

            # Compute delta and effect size
            mean_present = statistics.mean(present_acts)
            var_present = (
                statistics.variance(present_acts) if n_present > 1 else 0
            )

            sum_present = sum(present_acts)
            mean_absent = (
                (sum_all - sum_present) / n_absent if n_absent > 0 else 0
            )

            # Compute absent variance
            sum_sq_all = sum(a * a for a in all_activations)
            sum_sq_present = sum(a * a for a in present_acts)
            sum_sq_absent = sum_sq_all - sum_sq_present

            if n_absent > 1:
                mean_sq_absent = sum_sq_absent / n_absent
                var_absent = max(0, mean_sq_absent - mean_absent * mean_absent)
            else:
                var_absent = 0

            delta = mean_present - mean_absent

            pooled_var = (var_present * n_present + var_absent * n_absent) / (
                n_present + n_absent
            )
            pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
            effect_size = delta / pooled_std

            stat = {
                "token": token,
                "high_rate_ratio": round(high_rate_ratio, 4),
                "high_rate": round(high_rate, 4),
                "low_rate": round(low_rate, 4),
                "delta": round(delta, 6),
                "effect_size": round(effect_size, 4),
                "mean_present": round(mean_present, 6),
                "mean_absent": round(mean_absent, 6),
                "n_present": n_present,
                "n_absent": n_absent,
                "n_high_with_token": n_high_with_token,
            }
            token_stats.append(stat)

            # Classify as suppressor or enhancer
            if high_rate_ratio < suppressor_threshold:
                suppressors.append(stat)
            elif high_rate_ratio > enhancer_threshold:
                enhancers.append(stat)

        # Sort tables
        token_stats.sort(key=lambda x: x["high_rate_ratio"])
        suppressors.sort(key=lambda x: x["high_rate_ratio"])
        enhancers.sort(key=lambda x: -x["high_rate_ratio"])

        # Add tables to result
        result.add_table(
            "all_tokens",
            token_stats,
            columns=[
                "token",
                "high_rate_ratio",
                "high_rate",
                "low_rate",
                "delta",
                "effect_size",
                "n_present",
                "n_high_with_token",
            ],
            description="All token influence statistics sorted by high_rate_ratio",
        )

        result.add_table(
            "suppressors",
            suppressors,
            columns=[
                "token",
                "high_rate_ratio",
                "low_rate",
                "n_present",
                "n_high_with_token",
            ],
            description=f"Tokens with high_rate_ratio < {suppressor_threshold} (suppressed in high activation)",
        )

        result.add_table(
            "enhancers",
            enhancers,
            columns=[
                "token",
                "high_rate_ratio",
                "high_rate",
                "n_present",
                "n_high_with_token",
            ],
            description=f"Tokens with high_rate_ratio > {enhancer_threshold} (enriched in high activation)",
        )

        # Compute aggregates
        result.aggregates.n_samples = n_total
        result.aggregates.n_conditions = len(token_stats)
        result.aggregates.custom["n_high_examples"] = n_high
        result.aggregates.custom["n_low_examples"] = n_low
        result.aggregates.custom["high_threshold"] = round(high_threshold, 4)
        result.aggregates.custom["n_suppressors"] = len(suppressors)
        result.aggregates.custom["n_enhancers"] = len(enhancers)

        if suppressors:
            result.aggregates.custom["top_suppressor"] = suppressors[0]["token"]
            result.aggregates.custom["top_suppressor_ratio"] = suppressors[0][
                "high_rate_ratio"
            ]
        if enhancers:
            result.aggregates.custom["top_enhancer"] = enhancers[0]["token"]
            result.aggregates.custom["top_enhancer_ratio"] = enhancers[0][
                "high_rate_ratio"
            ]

        # Diagnostics
        result.diagnostics.n_contexts_tested = n_total

        # Check ReLU floor
        floor_detected, floor_rate = self._check_relu_floor(all_activations)
        result.diagnostics.relu_floor_detected = floor_detected
        result.diagnostics.relu_floor_rate = round(floor_rate, 3)
        if floor_detected:
            result.diagnostics.warnings.append(
                f"ReLU floor detected ({floor_rate:.1%} of examples)"
            )

        logger.info(
            f"Token influence sweep complete: {len(token_stats)} tokens analyzed, "
            f"{len(suppressors)} suppressors, {len(enhancers)} enhancers"
        )
