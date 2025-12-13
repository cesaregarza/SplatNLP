"""Family sweep experiment runners.

This module implements sweep experiments that test how ability families
affect feature activation across different AP rungs.
"""

import logging
import statistics
from collections import defaultdict
from typing import Any, Union

import polars as pl

from splatnlp.mechinterp.experiments.base import (
    ExperimentRunner,
    register_runner,
)
from splatnlp.mechinterp.schemas.experiment_results import (
    Aggregates,
    ExperimentResult,
    ResultTable,
)
from splatnlp.mechinterp.schemas.experiment_specs import (
    ExperimentSpec,
    ExperimentType,
    FamilySweepVariables,
)
from splatnlp.mechinterp.schemas.glossary import (
    STANDARD_RUNGS,
    TOKEN_FAMILIES,
    parse_token,
)
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


def _df_to_records(df: pl.DataFrame, ctx: MechInterpContext) -> list[dict]:
    """Convert Polars DataFrame to list of dicts with token names."""
    records = []
    for row in df.iter_rows(named=True):
        activation = row.get("activation", 0)
        token_ids = row.get("ability_input_tokens", [])
        weapon_id = row.get("weapon_id", 0)

        # Convert token IDs to names
        tokens = [ctx.inv_vocab.get(tid, "") for tid in token_ids]
        tokens = [t for t in tokens if t and not t.startswith("<")]

        records.append({
            "activation": activation,
            "tokens": tokens,
            "weapon_id": weapon_id,
        })
    return records


@register_runner
class Family1DSweepRunner(ExperimentRunner):
    """Runner for 1D family sweep experiments.

    Tests how a single ability family affects feature activation
    across different AP rungs.
    """

    name = "family_1d_sweep"
    handles_types = [ExperimentType.FAMILY_1D_SWEEP]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute family 1D sweep analysis."""
        # Parse variables
        vars_dict = spec.variables
        family = vars_dict.get("family")
        if not family:
            raise ValueError("Missing 'family' in variables")

        rungs = vars_dict.get("rungs") or STANDARD_RUNGS
        include_absent = vars_dict.get("include_absent", True)

        # Validate family exists
        if family not in TOKEN_FAMILIES:
            raise ValueError(f"Unknown family: {family}")

        family_info = TOKEN_FAMILIES[family]
        logger.info(f"Analyzing family {family} ({family_info.short_code})")

        # Get activation data
        logger.info("Loading activation data...")
        try:
            df = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            # Fallback for databases without this method
            df = ctx.db.get_feature_activations(spec.feature_id, limit=10000)

        if df is None or len(df) == 0:
            raise ValueError(
                f"No activation data for feature {spec.feature_id}"
            )

        logger.info(f"Loaded {len(df)} examples")

        # Convert DataFrame to records and apply slice
        data = self._apply_slice_df(df, spec, ctx)
        logger.info(f"After slicing: {len(data)} examples")

        # Group examples by family presence and rung
        groups = self._group_by_family(data, family, rungs, ctx)

        # Compute statistics
        rung_stats = []
        all_deltas = []

        # Baseline: examples without the family
        absent_acts = groups.get("absent", [])
        if absent_acts:
            baseline_mean = statistics.mean(absent_acts)
            baseline_std = (
                statistics.stdev(absent_acts) if len(absent_acts) > 1 else 0
            )
        else:
            baseline_mean = 0
            baseline_std = 0
            result.diagnostics.warnings.append(
                "No examples without family for baseline"
            )

        if include_absent:
            rung_stats.append(
                {
                    "rung": 0,
                    "label": "absent",
                    "mean_activation": round(baseline_mean, 4),
                    "std": round(baseline_std, 4),
                    "n": len(absent_acts),
                    "delta_from_baseline": 0.0,
                }
            )

        # Process each rung
        for rung in rungs:
            acts = groups.get(rung, [])
            if not acts:
                continue

            mean_act = statistics.mean(acts)
            std_act = statistics.stdev(acts) if len(acts) > 1 else 0
            delta = mean_act - baseline_mean

            rung_stats.append(
                {
                    "rung": rung,
                    "label": f"{family_info.short_code}_{rung}",
                    "mean_activation": round(mean_act, 4),
                    "std": round(std_act, 4),
                    "n": len(acts),
                    "delta_from_baseline": round(delta, 4),
                }
            )

            if len(acts) > 0:
                all_deltas.extend([mean_act - baseline_mean] * len(acts))

        # Populate result
        result.add_table(
            "rung_stats",
            rung_stats,
            columns=[
                "rung",
                "label",
                "mean_activation",
                "std",
                "n",
                "delta_from_baseline",
            ],
            description=f"Activation statistics by {family} AP rung",
        )

        # Aggregates
        if all_deltas:
            result.aggregates.mean_delta = round(statistics.mean(all_deltas), 4)
            result.aggregates.std_delta = round(
                statistics.stdev(all_deltas) if len(all_deltas) > 1 else 0, 4
            )
            result.aggregates.max_delta = round(max(all_deltas), 4)
            result.aggregates.min_delta = round(min(all_deltas), 4)
        else:
            result.aggregates.mean_delta = 0.0

        result.aggregates.base_activation_mean = round(baseline_mean, 4)
        result.aggregates.base_activation_std = round(baseline_std, 4)
        result.aggregates.n_samples = sum(s["n"] for s in rung_stats)
        result.aggregates.n_conditions = len(rung_stats)

        # Find threshold rung (where delta becomes significant)
        threshold_rung = self._find_threshold_rung(rung_stats)
        if threshold_rung:
            result.aggregates.custom["threshold_rung"] = threshold_rung

        # Diagnostics
        result.diagnostics.n_contexts_tested = len(data)

        # Check ReLU floor
        all_acts = [row.get("activation", 0) for row in data]
        floor_detected, floor_rate = self._check_relu_floor(all_acts)
        result.diagnostics.relu_floor_detected = floor_detected
        result.diagnostics.relu_floor_rate = round(floor_rate, 3)
        if floor_detected:
            result.diagnostics.warnings.append(
                f"ReLU floor detected ({floor_rate:.1%} of examples)"
            )

        logger.info(
            f"Sweep complete: {len(rung_stats)} conditions, "
            f"mean_delta={result.aggregates.mean_delta}"
        )

    def _apply_slice_df(
        self,
        df: pl.DataFrame,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
    ) -> list[dict]:
        """Apply dataset slicing to DataFrame and convert to records."""
        slice_cfg = spec.dataset_slice

        # Sort by activation for percentile filtering
        df = df.sort("activation")

        # Apply percentile filter
        if slice_cfg.percentile_min > 0 or slice_cfg.percentile_max < 100:
            n = len(df)
            start_idx = int(n * slice_cfg.percentile_min / 100)
            end_idx = int(n * slice_cfg.percentile_max / 100)
            df = df.slice(start_idx, end_idx - start_idx)

        # Apply sample size limit
        if slice_cfg.sample_size and len(df) > slice_cfg.sample_size:
            df = df.sample(
                n=slice_cfg.sample_size,
                seed=slice_cfg.random_seed or 42,
            )

        # Convert to records
        return _df_to_records(df, ctx)

    def _group_by_family(
        self,
        data: list[dict],
        family: str,
        rungs: list[int],
        ctx: MechInterpContext,
    ) -> dict[int | str, list[float]]:
        """Group examples by family presence and rung.

        Returns:
            Dict mapping rung (int) or 'absent' to list of activations
        """
        groups: dict[int | str, list[float]] = defaultdict(list)

        for row in data:
            activation = row.get("activation", 0)
            tokens = row.get("tokens", [])

            # Convert token IDs to names if needed
            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            # Find which rung of this family is present
            found_rung = None
            for token in tokens:
                token_family, ap = parse_token(token)
                if token_family == family and ap in rungs:
                    found_rung = ap
                    break

            if found_rung is not None:
                groups[found_rung].append(activation)
            else:
                groups["absent"].append(activation)

        return groups

    def _find_threshold_rung(
        self,
        rung_stats: list[dict],
        delta_threshold: float = 0.1,
    ) -> int | None:
        """Find the first rung where delta exceeds threshold."""
        for stat in rung_stats:
            if stat["rung"] > 0:  # Skip 'absent'
                if abs(stat["delta_from_baseline"]) > delta_threshold:
                    return stat["rung"]
        return None


@register_runner
class Family2DHeatmapRunner(ExperimentRunner):
    """Runner for 2D family heatmap experiments.

    Tests how two ability families interact by creating a 2D grid
    of combinations.
    """

    name = "family_2d_heatmap"
    handles_types = [ExperimentType.FAMILY_2D_HEATMAP]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute family 2D heatmap analysis."""
        # Parse variables
        vars_dict = spec.variables
        family_x = vars_dict.get("family_x")
        family_y = vars_dict.get("family_y")

        if not family_x or not family_y:
            raise ValueError("Missing 'family_x' or 'family_y' in variables")

        rungs_x = vars_dict.get("rungs_x") or STANDARD_RUNGS
        rungs_y = vars_dict.get("rungs_y") or STANDARD_RUNGS

        # Validate families
        for fam in [family_x, family_y]:
            if fam not in TOKEN_FAMILIES:
                raise ValueError(f"Unknown family: {fam}")

        family_x_info = TOKEN_FAMILIES[family_x]
        family_y_info = TOKEN_FAMILIES[family_y]

        logger.info(
            f"Analyzing {family_x_info.short_code} × {family_y_info.short_code}"
        )

        # Get activation data
        try:
            data = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            data = ctx.db.get_feature_activations(spec.feature_id, limit=10000)

        if data is None or len(data) == 0:
            raise ValueError(
                f"No activation data for feature {spec.feature_id}"
            )

        # Apply slice
        data = self._apply_slice(data, spec, ctx)
        logger.info(f"Processing {len(data)} examples")

        # Build 2D grid
        grid: dict[tuple[int, int], list[float]] = defaultdict(list)

        for row in data:
            activation = row.get("activation", 0)
            tokens = row.get("tokens", [])

            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            # Find rungs for both families
            rung_x, rung_y = 0, 0  # 0 = absent
            for token in tokens:
                token_family, ap = parse_token(token)
                if token_family == family_x and ap is not None:
                    rung_x = ap
                elif token_family == family_y and ap is not None:
                    rung_y = ap

            grid[(rung_x, rung_y)].append(activation)

        # Convert to table format
        heatmap_rows = []
        for (rx, ry), acts in sorted(grid.items()):
            if not acts:
                continue
            mean_act = statistics.mean(acts)
            heatmap_rows.append(
                {
                    f"{family_x_info.short_code}_rung": rx,
                    f"{family_y_info.short_code}_rung": ry,
                    "mean_activation": round(mean_act, 4),
                    "std": round(
                        statistics.stdev(acts) if len(acts) > 1 else 0, 4
                    ),
                    "n": len(acts),
                }
            )

        result.add_table(
            "heatmap",
            heatmap_rows,
            description=f"2D activation heatmap: {family_x_info.short_code} × {family_y_info.short_code}",
        )

        # Compute interaction strength
        # Compare diagonal to corners to estimate synergy
        baseline = grid.get((0, 0), [0])
        baseline_mean = statistics.mean(baseline) if baseline else 0

        all_deltas = []
        for (rx, ry), acts in grid.items():
            if rx > 0 or ry > 0:
                delta = statistics.mean(acts) - baseline_mean
                all_deltas.extend([delta] * len(acts))

        if all_deltas:
            result.aggregates.mean_delta = round(statistics.mean(all_deltas), 4)
            result.aggregates.max_delta = round(max(all_deltas), 4)

        result.aggregates.base_activation_mean = round(baseline_mean, 4)
        result.aggregates.n_samples = len(data)
        result.aggregates.n_conditions = len(heatmap_rows)

        # Find peak cell
        if heatmap_rows:
            peak = max(heatmap_rows, key=lambda x: x["mean_activation"])
            result.aggregates.custom["peak_x"] = peak[
                f"{family_x_info.short_code}_rung"
            ]
            result.aggregates.custom["peak_y"] = peak[
                f"{family_y_info.short_code}_rung"
            ]
            result.aggregates.custom["peak_activation"] = peak[
                "mean_activation"
            ]

        result.diagnostics.n_contexts_tested = len(data)

    def _apply_slice(self, df, spec, ctx):
        """Apply dataset slicing to DataFrame and convert to records.

        Reuses the logic from Family1DSweepRunner._apply_slice_df.
        """
        slice_cfg = spec.dataset_slice

        # Sort by activation for percentile filtering
        df = df.sort("activation")

        # Apply percentile filter
        if slice_cfg.percentile_min > 0 or slice_cfg.percentile_max < 100:
            n = len(df)
            start_idx = int(n * slice_cfg.percentile_min / 100)
            end_idx = int(n * slice_cfg.percentile_max / 100)
            df = df.slice(start_idx, end_idx - start_idx)

        # Apply sample size limit
        if slice_cfg.sample_size and len(df) > slice_cfg.sample_size:
            df = df.sample(
                n=slice_cfg.sample_size,
                seed=slice_cfg.random_seed or 42,
            )

        # Convert to records
        return _df_to_records(df, ctx)
