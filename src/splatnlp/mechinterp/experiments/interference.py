"""Interference analysis experiment runners.

This module implements analysis to detect when the presence of one token
REDUCES the activation caused by another token - an "interference" or
"error correction" effect.

This is particularly important for within-family effects, e.g.:
- SCU_3 reducing the signal from SCU_15
- Low AP rungs interfering with high AP rung detection
"""

import logging
import statistics
from collections import defaultdict
from itertools import product
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

        tokens = [ctx.inv_vocab.get(tid, "") for tid in token_ids]
        tokens = [t for t in tokens if t and not t.startswith("<")]

        records.append(
            {
                "activation": activation,
                "tokens": tokens,
                "weapon_id": weapon_id,
            }
        )
    return records


@register_runner
class WithinFamilyInterferenceRunner(ExperimentRunner):
    """Runner for within-family interference analysis.

    Detects when lower AP rungs of a family interfere with the
    activation caused by higher AP rungs.

    For example, if SCU_15 alone gives activation 0.12, but
    SCU_15 + SCU_3 gives activation 0.08, then SCU_3 is
    "interfering" with the SCU_15 signal.
    """

    name = "within_family_interference"
    handles_types = [ExperimentType.WITHIN_FAMILY_INTERFERENCE]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute within-family interference analysis."""
        vars_dict = spec.variables
        family = vars_dict.get("family")
        if not family:
            # Use modulator_token's family if specified
            modulator = vars_dict.get("modulator_token", "")
            fam, _ = parse_token(modulator)
            family = fam or "special_charge_up"  # Default

        rungs = vars_dict.get("rungs") or STANDARD_RUNGS

        if family not in TOKEN_FAMILIES:
            raise ValueError(f"Unknown family: {family}")

        family_info = TOKEN_FAMILIES[family]
        logger.info(f"Analyzing interference within {family_info.short_code}")

        # Get activation data
        try:
            df = ctx.db.get_all_feature_activations_for_pagerank(
                spec.feature_id
            )
        except AttributeError:
            df = ctx.db.get_feature_activations(spec.feature_id, limit=10000)

        if df is None or len(df) == 0:
            raise ValueError(f"No data for feature {spec.feature_id}")

        # Apply slice
        data = self._apply_slice(df, spec, ctx)
        logger.info(f"Processing {len(data)} examples")

        # Parse family tokens for each example
        parsed_data = []
        for row in data:
            tokens = row["tokens"]
            family_tokens = []
            family_rungs = []
            for t in tokens:
                tok_fam, ap = parse_token(t)
                if tok_fam == family and ap is not None:
                    family_tokens.append(t)
                    family_rungs.append(ap)

            parsed_data.append(
                {
                    "activation": row["activation"],
                    "weapon_id": row["weapon_id"],
                    "all_tokens": tokens,
                    "family_tokens": family_tokens,
                    "family_rungs": sorted(set(family_rungs)),
                }
            )

        # Analyze interference patterns
        # For each pair (low_rung, high_rung) where low < high:
        #   Compare: high alone vs high + low
        interference_results = []

        for low_rung in rungs:
            for high_rung in rungs:
                if low_rung >= high_rung:
                    continue

                # Find examples with:
                # 1. high_rung alone (no low_rung)
                # 2. high_rung + low_rung together

                high_alone = [
                    ex
                    for ex in parsed_data
                    if high_rung in ex["family_rungs"]
                    and low_rung not in ex["family_rungs"]
                ]

                high_with_low = [
                    ex
                    for ex in parsed_data
                    if high_rung in ex["family_rungs"]
                    and low_rung in ex["family_rungs"]
                ]

                if len(high_alone) < 5 or len(high_with_low) < 5:
                    continue

                mean_alone = statistics.mean(
                    [ex["activation"] for ex in high_alone]
                )
                mean_with_low = statistics.mean(
                    [ex["activation"] for ex in high_with_low]
                )
                delta = mean_with_low - mean_alone

                interference_results.append(
                    {
                        "high_rung": high_rung,
                        "low_rung": low_rung,
                        "high_alone_mean": round(mean_alone, 4),
                        "high_with_low_mean": round(mean_with_low, 4),
                        "delta": round(delta, 4),
                        "n_alone": len(high_alone),
                        "n_with_low": len(high_with_low),
                        "interference_type": (
                            "reduction"
                            if delta < -0.01
                            else ("enhancement" if delta > 0.01 else "neutral")
                        ),
                    }
                )

        # Sort by delta magnitude (most negative = strongest interference)
        interference_results.sort(key=lambda x: x["delta"])

        result.add_table(
            "interference_matrix",
            interference_results,
            columns=[
                "high_rung",
                "low_rung",
                "high_alone_mean",
                "high_with_low_mean",
                "delta",
                "n_alone",
                "n_with_low",
                "interference_type",
            ],
            description=f"Interference analysis: how low rungs affect high rungs in {family_info.short_code}",
        )

        # Find strongest interference
        reductions = [r for r in interference_results if r["delta"] < -0.01]
        enhancements = [r for r in interference_results if r["delta"] > 0.01]

        if reductions:
            strongest = min(reductions, key=lambda x: x["delta"])
            result.aggregates.custom["strongest_interference"] = {
                "low_rung": strongest["low_rung"],
                "high_rung": strongest["high_rung"],
                "delta": strongest["delta"],
            }

        if enhancements:
            strongest_enh = max(enhancements, key=lambda x: x["delta"])
            result.aggregates.custom["strongest_enhancement"] = {
                "low_rung": strongest_enh["low_rung"],
                "high_rung": strongest_enh["high_rung"],
                "delta": strongest_enh["delta"],
            }

        # Also analyze by weapon
        # Check if interference pattern is consistent across weapons
        weapon_interference = self._analyze_weapon_interference(
            parsed_data, family, rungs, ctx
        )
        if weapon_interference:
            result.add_table(
                "weapon_interference",
                weapon_interference[:20],
                columns=[
                    "weapon",
                    "high_rung",
                    "low_rung",
                    "delta",
                    "n",
                ],
                description="Interference patterns by weapon",
            )

        # Sample examples showing interference
        sample_examples = self._get_interference_examples(
            parsed_data, family_info.short_code, reductions, ctx
        )
        if sample_examples:
            result.add_table(
                "interference_examples",
                sample_examples,
                columns=["type", "activation", "family_tokens", "weapon"],
                description="Sample examples showing interference effect",
            )

        # Aggregates
        result.aggregates.n_samples = len(data)
        result.aggregates.n_conditions = len(interference_results)
        result.aggregates.custom["n_reductions"] = len(reductions)
        result.aggregates.custom["n_enhancements"] = len(enhancements)
        result.diagnostics.n_contexts_tested = len(data)

        logger.info(
            f"Found {len(reductions)} interference patterns, "
            f"{len(enhancements)} enhancement patterns"
        )

    def _analyze_weapon_interference(
        self,
        parsed_data: list[dict],
        family: str,
        rungs: list[int],
        ctx: MechInterpContext,
    ) -> list[dict]:
        """Analyze interference patterns broken down by weapon."""
        # Group by weapon
        by_weapon: dict[int, list[dict]] = defaultdict(list)
        for ex in parsed_data:
            by_weapon[ex["weapon_id"]].append(ex)

        results = []
        for weapon_id, examples in by_weapon.items():
            if len(examples) < 20:  # Need enough data per weapon
                continue

            weapon_name = ctx.id_to_weapon_display_name(weapon_id)

            # Check key rung pairs
            for low_rung, high_rung in [(3, 15), (3, 29), (6, 29)]:
                if low_rung not in rungs or high_rung not in rungs:
                    continue

                high_alone = [
                    ex
                    for ex in examples
                    if high_rung in ex["family_rungs"]
                    and low_rung not in ex["family_rungs"]
                ]
                high_with_low = [
                    ex
                    for ex in examples
                    if high_rung in ex["family_rungs"]
                    and low_rung in ex["family_rungs"]
                ]

                if len(high_alone) < 3 or len(high_with_low) < 3:
                    continue

                mean_alone = statistics.mean(
                    [ex["activation"] for ex in high_alone]
                )
                mean_with = statistics.mean(
                    [ex["activation"] for ex in high_with_low]
                )
                delta = mean_with - mean_alone

                if abs(delta) > 0.01:  # Only report significant effects
                    results.append(
                        {
                            "weapon": weapon_name,
                            "high_rung": high_rung,
                            "low_rung": low_rung,
                            "delta": round(delta, 4),
                            "n": len(high_alone) + len(high_with_low),
                        }
                    )

        results.sort(key=lambda x: x["delta"])
        return results

    def _get_interference_examples(
        self,
        parsed_data: list[dict],
        family_short: str,
        reductions: list[dict],
        ctx: MechInterpContext,
    ) -> list[dict]:
        """Get sample examples showing interference effect."""
        if not reductions:
            return []

        # Use the strongest reduction pair
        strongest = min(reductions, key=lambda x: x["delta"])
        high_rung = strongest["high_rung"]
        low_rung = strongest["low_rung"]

        examples = []

        # Get examples with high alone
        high_alone = [
            ex
            for ex in parsed_data
            if high_rung in ex["family_rungs"]
            and low_rung not in ex["family_rungs"]
        ]
        for ex in sorted(high_alone, key=lambda x: -x["activation"])[:3]:
            weapon = ctx.id_to_weapon_display_name(ex["weapon_id"])
            examples.append(
                {
                    "type": f"{family_short}_{high_rung} alone",
                    "activation": round(ex["activation"], 4),
                    "family_tokens": ", ".join(ex["family_tokens"]),
                    "weapon": weapon,
                }
            )

        # Get examples with high + low
        high_with_low = [
            ex
            for ex in parsed_data
            if high_rung in ex["family_rungs"]
            and low_rung in ex["family_rungs"]
        ]
        for ex in sorted(high_with_low, key=lambda x: -x["activation"])[:3]:
            weapon = ctx.id_to_weapon_display_name(ex["weapon_id"])
            examples.append(
                {
                    "type": f"{family_short}_{high_rung} + {family_short}_{low_rung}",
                    "activation": round(ex["activation"], 4),
                    "family_tokens": ", ".join(ex["family_tokens"]),
                    "weapon": weapon,
                }
            )

        return examples

    def _apply_slice(self, df, spec, ctx):
        """Apply dataset slicing and convert to records."""
        slice_cfg = spec.dataset_slice
        df = df.sort("activation")

        if slice_cfg.percentile_min > 0 or slice_cfg.percentile_max < 100:
            n = len(df)
            start_idx = int(n * slice_cfg.percentile_min / 100)
            end_idx = int(n * slice_cfg.percentile_max / 100)
            df = df.slice(start_idx, end_idx - start_idx)

        if slice_cfg.sample_size and len(df) > slice_cfg.sample_size:
            df = df.sample(
                n=slice_cfg.sample_size, seed=slice_cfg.random_seed or 42
            )

        return _df_to_records(df, ctx)
