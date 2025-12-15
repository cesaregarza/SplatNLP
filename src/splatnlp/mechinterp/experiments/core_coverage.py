"""Core coverage analysis experiment runner.

This module implements the core_coverage_analysis experiment which checks
if tokens are tail markers vs primary drivers by computing coverage in
the core region (25-75% of effective max).

The key insight: tokens with high enrichment in tail examples but low
coverage in core examples are "tail markers" (super-stimuli), not the
headline concept of the feature.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np

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
class CoreCoverageRunner(ExperimentRunner):
    """Runner for core coverage analysis experiments.

    Computes coverage metrics to distinguish tail markers from primary drivers:
    - core_coverage: % of core examples containing the token
    - tail_enrichment: How much more the token appears in tail vs core
    - is_tail_marker: Whether core_coverage < threshold

    This helps avoid ability/weapon flanderization where high-activation
    examples are dominated by a token that is actually rare in the core mass.
    """

    name = "core_coverage_analysis"
    handles_types = [ExperimentType.CORE_COVERAGE_ANALYSIS]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute core coverage analysis."""
        # Parse variables
        vars_dict = spec.variables
        tokens_to_check = vars_dict.get("tokens_to_check", None)
        top_k = vars_dict.get("top_k", 10)
        coverage_threshold = vars_dict.get("coverage_threshold", 0.30)
        include_weapons = vars_dict.get("include_weapons", True)
        core_lower_pct = vars_dict.get("core_lower_pct", 0.25)
        core_upper_pct = vars_dict.get("core_upper_pct", 0.75)
        effective_max_percentile = vars_dict.get(
            "effective_max_percentile", 0.995
        )

        logger.info(
            f"Running core coverage analysis "
            f"(core={core_lower_pct:.0%}-{core_upper_pct:.0%}, "
            f"threshold={coverage_threshold:.0%})"
        )

        # Get activation data
        logger.info("Loading activation data...")
        try:
            df = ctx.db.get_all_feature_activations_for_pagerank(spec.feature_id)
        except AttributeError:
            df = ctx.db.get_feature_activations(spec.feature_id, limit=100000)

        if df is None or len(df) == 0:
            raise ValueError(f"No activation data for feature {spec.feature_id}")

        logger.info(f"Loaded {len(df)} examples")

        # Extract activation values
        activations = np.array(df["activation"].to_list())
        nonzero_mask = activations > 0
        nonzero_acts = activations[nonzero_mask]

        if len(nonzero_acts) == 0:
            raise ValueError("No nonzero activations found")

        # Compute effective max (99.5th percentile to avoid outliers)
        effective_max = float(
            np.percentile(nonzero_acts, effective_max_percentile * 100)
        )

        # Define regions
        core_lower = effective_max * core_lower_pct
        core_upper = effective_max * core_upper_pct
        tail_lower = effective_max * 0.90  # Flanderization zone: 90%+ of max

        # Create region masks
        core_mask = (activations > core_lower) & (activations <= core_upper)
        tail_mask = activations > tail_lower

        n_total = len(activations)
        n_core = int(np.sum(core_mask))
        n_tail = int(np.sum(tail_mask))

        logger.info(
            f"Effective max: {effective_max:.4f}, "
            f"Core: {n_core} examples ({n_core/n_total:.1%}), "
            f"Tail: {n_tail} examples ({n_tail/n_total:.1%})"
        )

        # If no tokens specified, use PageRank top tokens
        if tokens_to_check is None:
            tokens_to_check = self._get_top_pagerank_tokens(
                df, ctx, top_k, collapse_families=True
            )
            logger.info(f"Using top {len(tokens_to_check)} PageRank tokens")

        # Compute token coverage in core vs tail
        token_coverage_stats: list[dict[str, Any]] = []

        for token in tokens_to_check:
            stats = self._compute_token_coverage(
                df,
                ctx,
                token,
                core_mask,
                tail_mask,
                n_core,
                n_tail,
                coverage_threshold,
            )
            if stats:
                token_coverage_stats.append(stats)

        # Sort by core coverage (lowest first to highlight tail markers)
        token_coverage_stats.sort(key=lambda x: x["core_coverage_pct"])

        # Add token coverage table
        result.add_table(
            "token_coverage",
            token_coverage_stats,
            columns=[
                "token",
                "core_coverage_pct",
                "tail_coverage_pct",
                "tail_enrichment",
                "is_tail_marker",
                "n_core_with_token",
                "n_tail_with_token",
            ],
            description=(
                f"Token coverage in core ({core_lower_pct:.0%}-{core_upper_pct:.0%}) "
                f"vs tail (90%+) regions. Tail markers have <{coverage_threshold:.0%} core coverage."
            ),
        )

        # Compute weapon coverage if requested
        if include_weapons:
            weapon_coverage_stats = self._compute_weapon_coverage(
                df,
                ctx,
                core_mask,
                tail_mask,
                n_core,
                n_tail,
                top_k=top_k,
            )

            result.add_table(
                "weapon_coverage",
                weapon_coverage_stats,
                columns=[
                    "weapon",
                    "weapon_id",
                    "core_coverage_pct",
                    "tail_coverage_pct",
                    "tail_enrichment",
                    "is_flanderized",
                    "n_core",
                    "n_tail",
                ],
                description="Weapon coverage in core vs tail regions. Flanderized = dominant in tail but rare in core.",
            )

        # Compute aggregates
        tail_markers = [s for s in token_coverage_stats if s["is_tail_marker"]]
        primary_drivers = [
            s for s in token_coverage_stats if not s["is_tail_marker"]
        ]

        result.aggregates.n_samples = n_total
        result.aggregates.n_conditions = len(token_coverage_stats)
        result.aggregates.custom["effective_max"] = round(effective_max, 4)
        result.aggregates.custom["n_core_examples"] = n_core
        result.aggregates.custom["n_tail_examples"] = n_tail
        result.aggregates.custom["core_region"] = (
            f"{core_lower_pct:.0%}-{core_upper_pct:.0%}"
        )
        result.aggregates.custom["n_tail_markers"] = len(tail_markers)
        result.aggregates.custom["n_primary_drivers"] = len(primary_drivers)
        result.aggregates.custom["coverage_threshold"] = coverage_threshold

        if tail_markers:
            result.aggregates.custom["top_tail_marker"] = tail_markers[0]["token"]
            result.aggregates.custom["top_tail_marker_coverage"] = tail_markers[
                0
            ]["core_coverage_pct"]
            result.aggregates.custom["top_tail_marker_enrichment"] = tail_markers[
                0
            ]["tail_enrichment"]

        if primary_drivers:
            # Best primary driver = highest core coverage
            best_driver = max(primary_drivers, key=lambda x: x["core_coverage_pct"])
            result.aggregates.custom["best_primary_driver"] = best_driver["token"]
            result.aggregates.custom["best_driver_coverage"] = best_driver[
                "core_coverage_pct"
            ]

        # Diagnostics
        result.diagnostics.n_contexts_tested = n_total

        # Check ReLU floor
        floor_detected, floor_rate = self._check_relu_floor(
            activations.tolist()
        )
        result.diagnostics.relu_floor_detected = floor_detected
        result.diagnostics.relu_floor_rate = round(floor_rate, 3)
        if floor_detected:
            result.diagnostics.warnings.append(
                f"ReLU floor detected ({floor_rate:.1%} of examples)"
            )

        # Add warning if no primary drivers found
        if not primary_drivers and tail_markers:
            result.diagnostics.warnings.append(
                f"All {len(tail_markers)} checked tokens are tail markers! "
                "Feature may be diffuse or need different token candidates."
            )

        logger.info(
            f"Core coverage analysis complete: {len(primary_drivers)} primary drivers, "
            f"{len(tail_markers)} tail markers"
        )

    def _get_top_pagerank_tokens(
        self,
        df,
        ctx: MechInterpContext,
        top_k: int,
        collapse_families: bool = True,
    ) -> list[str]:
        """Get top PageRank tokens from activation data."""
        # Simple frequency-based approximation for top tokens
        token_counts: dict[str, int] = defaultdict(int)

        for row in df.iter_rows(named=True):
            token_ids = row.get("ability_input_tokens", [])
            activation = row.get("activation", 0)

            # Weight by activation
            weight = max(1.0, activation * 10)

            seen = set()
            for tid in token_ids or []:
                tname = ctx.inv_vocab.get(tid, "")
                if collapse_families:
                    base = _token_base(tname)
                else:
                    base = tname
                if base and base not in seen:
                    seen.add(base)
                    token_counts[base] += int(weight)

        # Sort by count and return top-k
        sorted_tokens = sorted(
            token_counts.items(), key=lambda x: -x[1]
        )
        return [t[0] for t in sorted_tokens[:top_k]]

    def _compute_token_coverage(
        self,
        df,
        ctx: MechInterpContext,
        token: str,
        core_mask: np.ndarray,
        tail_mask: np.ndarray,
        n_core: int,
        n_tail: int,
        coverage_threshold: float,
    ) -> dict[str, Any] | None:
        """Compute coverage statistics for a single token."""
        n_core_with_token = 0
        n_tail_with_token = 0

        for idx, row in enumerate(df.iter_rows(named=True)):
            token_ids = row.get("ability_input_tokens", [])

            # Check if token (or any member of family) is present
            has_token = False
            for tid in token_ids or []:
                tname = ctx.inv_vocab.get(tid, "")
                base = _token_base(tname)
                if base == token or tname == token:
                    has_token = True
                    break

            if has_token:
                if core_mask[idx]:
                    n_core_with_token += 1
                if tail_mask[idx]:
                    n_tail_with_token += 1

        if n_core == 0 or n_tail == 0:
            return None

        core_coverage = n_core_with_token / n_core
        tail_coverage = n_tail_with_token / n_tail

        # Tail enrichment = tail_coverage / core_coverage
        if core_coverage > 0.001:
            tail_enrichment = tail_coverage / core_coverage
        elif tail_coverage > 0:
            tail_enrichment = 10.0  # Cap at 10x
        else:
            tail_enrichment = 1.0

        is_tail_marker = core_coverage < coverage_threshold

        return {
            "token": token,
            "core_coverage_pct": round(core_coverage * 100, 1),
            "tail_coverage_pct": round(tail_coverage * 100, 1),
            "tail_enrichment": round(tail_enrichment, 2),
            "is_tail_marker": is_tail_marker,
            "n_core_with_token": n_core_with_token,
            "n_tail_with_token": n_tail_with_token,
        }

    def _compute_weapon_coverage(
        self,
        df,
        ctx: MechInterpContext,
        core_mask: np.ndarray,
        tail_mask: np.ndarray,
        n_core: int,
        n_tail: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Compute coverage statistics for weapons."""
        core_weapon_counts: dict[int, int] = defaultdict(int)
        tail_weapon_counts: dict[int, int] = defaultdict(int)

        for idx, row in enumerate(df.iter_rows(named=True)):
            weapon_id = row.get("weapon_id")
            if weapon_id is None:
                continue

            if core_mask[idx]:
                core_weapon_counts[weapon_id] += 1
            if tail_mask[idx]:
                tail_weapon_counts[weapon_id] += 1

        # Get top weapons by tail count (most likely to be flanderized)
        all_weapons = set(core_weapon_counts.keys()) | set(
            tail_weapon_counts.keys()
        )

        weapon_stats = []
        for weapon_id in all_weapons:
            n_core_weapon = core_weapon_counts.get(weapon_id, 0)
            n_tail_weapon = tail_weapon_counts.get(weapon_id, 0)

            if n_core == 0:
                continue

            core_coverage = n_core_weapon / n_core
            tail_coverage = n_tail_weapon / n_tail if n_tail > 0 else 0

            if core_coverage > 0.001:
                tail_enrichment = tail_coverage / core_coverage
            elif tail_coverage > 0:
                tail_enrichment = 10.0
            else:
                tail_enrichment = 1.0

            # Flanderized = dominant in tail but much less in core
            is_flanderized = tail_enrichment > 2.0 and tail_coverage > 0.10

            # Get weapon name
            weapon_name = ctx.inv_weapon_vocab.get(weapon_id, f"weapon_{weapon_id}")

            weapon_stats.append(
                {
                    "weapon": weapon_name,
                    "weapon_id": weapon_id,
                    "core_coverage_pct": round(core_coverage * 100, 1),
                    "tail_coverage_pct": round(tail_coverage * 100, 1),
                    "tail_enrichment": round(tail_enrichment, 2),
                    "is_flanderized": is_flanderized,
                    "n_core": n_core_weapon,
                    "n_tail": n_tail_weapon,
                }
            )

        # Sort by tail_enrichment (highest = most flanderized)
        weapon_stats.sort(key=lambda x: -x["tail_enrichment"])

        return weapon_stats[:top_k]
