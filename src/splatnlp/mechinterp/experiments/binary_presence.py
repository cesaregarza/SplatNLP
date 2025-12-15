"""Binary presence effect experiment runner.

This module implements the binary_presence_effect experiment which analyzes
how binary abilities (main-only abilities without AP rungs) affect feature
activation. Unlike scaling abilities, binary abilities are either present
or absent, making their analysis different from family sweeps.
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
from splatnlp.mechinterp.schemas.glossary import (
    CANONICAL_MAIN_ONLY_ABILITIES,
    parse_token,
)
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


# Binary abilities to analyze (main-only abilities)
BINARY_ABILITIES = list(CANONICAL_MAIN_ONLY_ABILITIES.keys())


@register_runner
class BinaryPresenceEffectRunner(ExperimentRunner):
    """Runner for binary presence effect experiments.

    Analyzes how binary abilities affect feature activation by computing:
    - mean_with: Mean activation when binary ability is present
    - mean_without: Mean activation when binary ability is absent
    - delta: Difference (mean_with - mean_without)
    - presence_enrichment: How often the ability appears in high-activation examples
      relative to baseline

    Optionally stratifies by a primary driver (scaling ability) to show
    conditional effects at each primary rung level.
    """

    name = "binary_presence_effect"
    handles_types = [ExperimentType.BINARY_PRESENCE_EFFECT]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute binary presence effect analysis."""
        # Parse variables
        vars_dict = spec.variables
        binary_tokens = vars_dict.get("binary_tokens", BINARY_ABILITIES)
        primary_family = vars_dict.get("primary_family", None)
        primary_rungs = vars_dict.get("primary_rungs", [0, 12, 29, 41, 57])
        min_samples = vars_dict.get("min_samples", 30)
        high_percentile = vars_dict.get("high_percentile", 0.90)

        logger.info(
            f"Running binary presence effect (primary={primary_family}, "
            f"n_binary={len(binary_tokens)})"
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

        # Build lookup for binary tokens
        binary_token_ids = set()
        for binary_name in binary_tokens:
            tid = ctx.vocab.get(binary_name)
            if tid is not None:
                binary_token_ids.add(tid)

        # Organize data
        all_activations: list[float] = []
        binary_presence: dict[str, list[tuple[float, dict]]] = defaultdict(list)

        for row in df.iter_rows(named=True):
            token_ids = row.get("ability_input_tokens", [])
            activation = row.get("activation", 0)
            all_activations.append(activation)

            # Check which binary abilities are present
            row_data = {
                "activation": activation,
                "tokens": token_ids,
            }

            # Parse primary family rung if specified
            if primary_family:
                primary_rung = 0  # Default: absent
                for tid in token_ids or []:
                    tname = ctx.inv_vocab.get(tid, "")
                    family_name, ap = parse_token(tname)
                    if family_name == primary_family and ap is not None:
                        primary_rung = ap
                        break
                row_data["primary_rung"] = primary_rung

            # Track binary ability presence
            for tid in token_ids or []:
                tname = ctx.inv_vocab.get(tid, "")
                if tname in binary_tokens:
                    binary_presence[tname].append((activation, row_data))

        n_total = len(all_activations)
        if n_total == 0:
            raise ValueError("No activations found")

        # Compute high activation threshold
        sorted_acts = sorted(all_activations, reverse=True)
        high_idx = max(1, int(n_total * (1 - high_percentile)))
        high_threshold = sorted_acts[min(high_idx, len(sorted_acts) - 1)]
        n_high = sum(1 for a in all_activations if a >= high_threshold)

        logger.info(
            f"High activation threshold: {high_threshold:.4f} (n_high={n_high})"
        )

        # Compute overall statistics for each binary ability
        overall_stats: list[dict[str, Any]] = []

        for binary_name in binary_tokens:
            present_acts = [a for a, _ in binary_presence.get(binary_name, [])]
            n_present = len(present_acts)
            n_absent = n_total - n_present

            if n_present < min_samples:
                continue

            # Mean with/without
            mean_with = statistics.mean(present_acts) if present_acts else 0
            sum_all = sum(all_activations)
            sum_present = sum(present_acts)
            mean_without = (
                (sum_all - sum_present) / n_absent if n_absent > 0 else 0
            )
            delta = mean_with - mean_without

            # Presence enrichment in high activation
            n_high_with = sum(1 for a in present_acts if a >= high_threshold)
            baseline_rate = n_present / n_total
            high_rate = n_high_with / n_high if n_high > 0 else 0
            enrichment = high_rate / baseline_rate if baseline_rate > 0 else 0

            # Effect size (Cohen's d)
            if n_present > 1 and n_absent > 1:
                var_with = statistics.variance(present_acts)
                # Compute absent variance
                sum_sq_all = sum(a * a for a in all_activations)
                sum_sq_present = sum(a * a for a in present_acts)
                sum_sq_absent = sum_sq_all - sum_sq_present
                mean_sq_absent = sum_sq_absent / n_absent
                var_without = max(0, mean_sq_absent - mean_without**2)

                pooled_var = (var_with * n_present + var_without * n_absent) / (
                    n_present + n_absent
                )
                pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
                effect_size = delta / pooled_std
            else:
                effect_size = 0

            stat = {
                "binary_ability": binary_name,
                "mean_with": round(mean_with, 6),
                "mean_without": round(mean_without, 6),
                "delta": round(delta, 6),
                "effect_size": round(effect_size, 4),
                "n_present": n_present,
                "n_absent": n_absent,
                "baseline_rate": round(baseline_rate, 4),
                "high_rate": round(high_rate, 4),
                "enrichment": round(enrichment, 4),
                "n_high_with": n_high_with,
            }
            overall_stats.append(stat)

        # Sort by absolute delta
        overall_stats.sort(key=lambda x: abs(x["delta"]), reverse=True)

        # Add overall stats table
        result.add_table(
            "binary_effects",
            overall_stats,
            columns=[
                "binary_ability",
                "delta",
                "effect_size",
                "enrichment",
                "mean_with",
                "mean_without",
                "n_present",
                "n_high_with",
            ],
            description="Binary ability presence effects sorted by |delta|",
        )

        # If primary family specified, compute conditional effects
        if primary_family:
            conditional_stats: list[dict[str, Any]] = []

            for binary_name in binary_tokens:
                present_data = binary_presence.get(binary_name, [])
                if len(present_data) < min_samples:
                    continue

                # Group by primary rung
                for rung in primary_rungs:
                    # Examples with this primary rung
                    with_binary_at_rung = [
                        a
                        for a, data in present_data
                        if data.get("primary_rung", 0) == rung
                    ]

                    # All examples at this rung (to compute without)
                    all_at_rung = [
                        activation
                        for activation, token_ids in zip(
                            all_activations, df.iter_rows(named=True)
                        )
                        for _ in [1]  # dummy to allow computation
                    ]
                    # Recompute: need to iterate properly
                    rung_activations = []
                    rung_with_binary = []
                    rung_without_binary = []

                    for row in df.iter_rows(named=True):
                        token_ids = row.get("ability_input_tokens", [])
                        activation = row.get("activation", 0)

                        # Check primary rung
                        row_rung = 0
                        has_binary = False
                        for tid in token_ids or []:
                            tname = ctx.inv_vocab.get(tid, "")
                            family_name, ap = parse_token(tname)
                            if family_name == primary_family and ap is not None:
                                row_rung = ap
                            if tname == binary_name:
                                has_binary = True

                        if row_rung == rung:
                            rung_activations.append(activation)
                            if has_binary:
                                rung_with_binary.append(activation)
                            else:
                                rung_without_binary.append(activation)

                    n_at_rung = len(rung_activations)
                    n_with = len(rung_with_binary)
                    n_without = len(rung_without_binary)

                    if n_with < 5 or n_without < 5:
                        continue

                    mean_with = statistics.mean(rung_with_binary)
                    mean_without = statistics.mean(rung_without_binary)
                    delta = mean_with - mean_without

                    conditional_stats.append(
                        {
                            "binary_ability": binary_name,
                            "primary_rung": rung,
                            "mean_with": round(mean_with, 6),
                            "mean_without": round(mean_without, 6),
                            "delta": round(delta, 6),
                            "n_with": n_with,
                            "n_without": n_without,
                            "n_at_rung": n_at_rung,
                        }
                    )

            # Sort by binary ability, then rung
            conditional_stats.sort(
                key=lambda x: (x["binary_ability"], x["primary_rung"])
            )

            result.add_table(
                "conditional_effects",
                conditional_stats,
                columns=[
                    "binary_ability",
                    "primary_rung",
                    "delta",
                    "mean_with",
                    "mean_without",
                    "n_with",
                    "n_without",
                ],
                description=f"Conditional effects of binary abilities at each {primary_family} rung",
            )

        # Aggregates
        result.aggregates.n_samples = n_total
        result.aggregates.n_conditions = len(overall_stats)
        result.aggregates.custom["n_high_examples"] = n_high
        result.aggregates.custom["high_threshold"] = round(high_threshold, 4)

        if overall_stats:
            top_effect = max(overall_stats, key=lambda x: abs(x["delta"]))
            result.aggregates.custom["top_effect_ability"] = top_effect[
                "binary_ability"
            ]
            result.aggregates.custom["top_effect_delta"] = top_effect["delta"]

            # Find most enriched and most depleted
            most_enriched = max(overall_stats, key=lambda x: x["enrichment"])
            most_depleted = min(overall_stats, key=lambda x: x["enrichment"])
            result.aggregates.custom["most_enriched"] = most_enriched[
                "binary_ability"
            ]
            result.aggregates.custom["most_enriched_ratio"] = most_enriched[
                "enrichment"
            ]
            result.aggregates.custom["most_depleted"] = most_depleted[
                "binary_ability"
            ]
            result.aggregates.custom["most_depleted_ratio"] = most_depleted[
                "enrichment"
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
            f"Binary presence effect complete: {len(overall_stats)} abilities analyzed"
        )
