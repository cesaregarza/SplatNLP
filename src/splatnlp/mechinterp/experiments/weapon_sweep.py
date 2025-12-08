"""Weapon sweep experiment runners.

This module implements weapon-based analysis to understand how features
respond to different weapons, weapon classes, and weapon-ability interactions.
"""

import logging
import statistics
from collections import defaultdict
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
from splatnlp.mechinterp.schemas.glossary import TOKEN_FAMILIES, parse_token
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


# Weapon class mappings (simplified - can be expanded)
WEAPON_CLASSES = {
    "shooter": [
        "Splattershot",
        "Splattershot Jr.",
        "N-ZAP '85",
        "N-ZAP '89",
        ".52 Gal",
        ".96 Gal",
        "Aerospray MG",
        "Aerospray RG",
        "Jet Squelcher",
        "L-3 Nozzlenose",
        "H-3 Nozzlenose",
        "Splash-o-matic",
        "Sploosh-o-matic",
        "Splattershot Pro",
        "Squeezer",
    ],
    "roller": [
        "Splat Roller",
        "Carbon Roller",
        "Dynamo Roller",
        "Flingza Roller",
        "Big Swig Roller",
    ],
    "charger": [
        "Splat Charger",
        "Splatterscope",
        "E-liter 4K",
        "E-liter 4K Scope",
        "Goo Tuber",
        "Bamboozler 14 Mk I",
        "Snipewriter 5H",
    ],
    "slosher": [
        "Slosher",
        "Tri-Slosher",
        "Sloshing Machine",
        "Bloblobber",
        "Explosher",
    ],
    "splatling": [
        "Mini Splatling",
        "Heavy Splatling",
        "Hydra Splatling",
        "Ballpoint Splatling",
        "Nautilus 47",
    ],
    "dualies": [
        "Splat Dualies",
        "Dapple Dualies",
        "Glooga Dualies",
        "Dualie Squelchers",
        "Dark Tetra Dualies",
    ],
    "brella": [
        "Splat Brella",
        "Tenta Brella",
        "Undercover Brella",
    ],
    "blaster": [
        "Blaster",
        "Range Blaster",
        "Luna Blaster",
        "Clash Blaster",
        "Rapid Blaster",
        "Rapid Blaster Pro",
        "S-BLAST '92",
    ],
    "brush": [
        "Inkbrush",
        "Octobrush",
        "Painbrush",
    ],
    "stringer": [
        "Tri-Stringer",
        "REEF-LUX 450",
    ],
    "splatana": [
        "Splatana Stamper",
        "Splatana Wiper",
    ],
}


def get_weapon_class(weapon_name: str) -> str | None:
    """Get the weapon class for a weapon name."""
    weapon_lower = weapon_name.lower()
    for cls, weapons in WEAPON_CLASSES.items():
        for w in weapons:
            if w.lower() in weapon_lower or weapon_lower in w.lower():
                return cls
    return None


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
class WeaponSweepRunner(ExperimentRunner):
    """Runner for weapon sweep experiments.

    Analyzes how a feature responds across different weapons,
    optionally conditioned on ability presence.
    """

    name = "weapon_sweep"
    handles_types = [ExperimentType.WEAPON_SWEEP]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute weapon sweep analysis."""
        vars_dict = spec.variables
        condition_family = vars_dict.get(
            "condition_family"
        )  # Optional: only consider examples with this family
        min_examples = vars_dict.get(
            "min_examples", 10
        )  # Min examples per weapon
        top_k_weapons = vars_dict.get("top_k_weapons", 20)  # Limit output

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

        # Group by weapon
        weapon_groups: dict[int, list[float]] = defaultdict(list)
        for row in data:
            tokens = row.get("tokens", [])

            # If conditioning on family, filter examples
            if condition_family:
                has_family = any(
                    parse_token(t)[0] == condition_family for t in tokens
                )
                if not has_family:
                    continue

            weapon_id = row.get("weapon_id", 0)
            activation = row.get("activation", 0)
            weapon_groups[weapon_id].append(activation)

        # Compute statistics per weapon
        weapon_stats = []
        all_acts = [row["activation"] for row in data]
        global_mean = statistics.mean(all_acts) if all_acts else 0

        for weapon_id, acts in weapon_groups.items():
            if len(acts) < min_examples:
                continue

            weapon_name = ctx.id_to_weapon_display_name(weapon_id)
            weapon_class = get_weapon_class(weapon_name)
            mean_act = statistics.mean(acts)
            std_act = statistics.stdev(acts) if len(acts) > 1 else 0
            delta = mean_act - global_mean

            weapon_stats.append(
                {
                    "weapon_id": weapon_id,
                    "weapon_name": weapon_name,
                    "weapon_class": weapon_class or "unknown",
                    "mean_activation": round(mean_act, 4),
                    "std": round(std_act, 4),
                    "n": len(acts),
                    "delta_from_global": round(delta, 4),
                }
            )

        # Sort by mean activation
        weapon_stats.sort(key=lambda x: x["mean_activation"], reverse=True)

        # Limit to top_k
        weapon_stats = weapon_stats[:top_k_weapons]

        result.add_table(
            "weapon_stats",
            weapon_stats,
            columns=[
                "weapon_name",
                "weapon_class",
                "mean_activation",
                "std",
                "n",
                "delta_from_global",
            ],
            description="Activation statistics by weapon",
        )

        # Aggregate by weapon class
        class_groups: dict[str, list[float]] = defaultdict(list)
        for row in data:
            weapon_id = row.get("weapon_id", 0)
            weapon_name = ctx.id_to_weapon_display_name(weapon_id)
            weapon_class = get_weapon_class(weapon_name)
            if weapon_class:
                class_groups[weapon_class].append(row["activation"])

        class_stats = []
        for cls, acts in sorted(class_groups.items()):
            if len(acts) < min_examples:
                continue
            mean_act = statistics.mean(acts)
            class_stats.append(
                {
                    "weapon_class": cls,
                    "mean_activation": round(mean_act, 4),
                    "std": round(
                        statistics.stdev(acts) if len(acts) > 1 else 0, 4
                    ),
                    "n": len(acts),
                    "delta_from_global": round(mean_act - global_mean, 4),
                }
            )

        class_stats.sort(key=lambda x: x["mean_activation"], reverse=True)

        result.add_table(
            "class_stats",
            class_stats,
            columns=[
                "weapon_class",
                "mean_activation",
                "std",
                "n",
                "delta_from_global",
            ],
            description="Activation statistics by weapon class",
        )

        # Populate aggregates
        if weapon_stats:
            top_weapon = weapon_stats[0]
            result.aggregates.custom["top_weapon"] = top_weapon["weapon_name"]
            result.aggregates.custom["top_weapon_mean"] = top_weapon[
                "mean_activation"
            ]
            result.aggregates.custom["top_weapon_delta"] = top_weapon[
                "delta_from_global"
            ]

            # Find weapon with lowest activation
            bottom_weapon = min(
                weapon_stats, key=lambda x: x["mean_activation"]
            )
            result.aggregates.custom["bottom_weapon"] = bottom_weapon[
                "weapon_name"
            ]
            result.aggregates.custom["weapon_range"] = round(
                top_weapon["mean_activation"]
                - bottom_weapon["mean_activation"],
                4,
            )

            # Check for dominant weapon (>2x delta of second weapon)
            if len(weapon_stats) >= 2:
                second_weapon = weapon_stats[1]
                top_delta = abs(top_weapon["delta_from_global"])
                second_delta = abs(second_weapon["delta_from_global"])

                if second_delta > 0 and top_delta > 2 * second_delta:
                    ratio = top_delta / second_delta
                    result.diagnostics.warnings.append(
                        f"DOMINANT WEAPON: {top_weapon['weapon_name']} has "
                        f"{ratio:.1f}x the delta of second weapon "
                        f"({second_weapon['weapon_name']}). "
                        f"Run kit_sweep to check sub/special patterns."
                    )
                    result.aggregates.custom["dominant_weapon_detected"] = True
                    result.aggregates.custom["dominant_weapon_ratio"] = round(
                        ratio, 2
                    )
                    result.aggregates.custom["recommended_followup"] = (
                        "kit_sweep"
                    )

        if class_stats:
            top_class = class_stats[0]
            result.aggregates.custom["top_class"] = top_class["weapon_class"]

        result.aggregates.base_activation_mean = round(global_mean, 4)
        result.aggregates.n_samples = len(data)
        result.aggregates.n_conditions = len(weapon_stats)
        result.diagnostics.n_contexts_tested = len(data)

        logger.info(
            f"Analyzed {len(weapon_stats)} weapons, {len(class_stats)} classes"
        )

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


@register_runner
class WeaponGroupAnalysisRunner(ExperimentRunner):
    """Runner for weapon group analysis.

    Compares high-activation vs low-activation examples by weapon distribution
    to identify weapon-specific effects.
    """

    name = "weapon_group_analysis"
    handles_types = [ExperimentType.WEAPON_GROUP_ANALYSIS]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute weapon group analysis."""
        vars_dict = spec.variables
        high_pct = vars_dict.get("high_percentile", 10)  # Top X%
        low_pct = vars_dict.get("low_percentile", 10)  # Bottom X%

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
        n = len(data)
        logger.info(f"Processing {n} examples")

        # Sort by activation
        sorted_data = sorted(data, key=lambda x: x["activation"], reverse=True)
        high_n = max(1, int(n * high_pct / 100))
        low_n = max(1, int(n * low_pct / 100))

        high_examples = sorted_data[:high_n]
        low_examples = sorted_data[-low_n:]

        # Count weapons in each group
        high_weapons = defaultdict(int)
        low_weapons = defaultdict(int)

        for ex in high_examples:
            w = ctx.id_to_weapon_display_name(ex["weapon_id"])
            high_weapons[w] += 1

        for ex in low_examples:
            w = ctx.id_to_weapon_display_name(ex["weapon_id"])
            low_weapons[w] += 1

        # Compute enrichment ratio
        all_weapons = set(high_weapons.keys()) | set(low_weapons.keys())
        enrichment = []
        for w in all_weapons:
            high_count = high_weapons.get(w, 0)
            low_count = low_weapons.get(w, 0)

            # Enrichment = (high_frac) / (low_frac)
            high_frac = high_count / high_n if high_n > 0 else 0
            low_frac = low_count / low_n if low_n > 0 else 0

            if low_frac > 0:
                ratio = high_frac / low_frac
            elif high_frac > 0:
                ratio = float("inf")
            else:
                ratio = 1.0

            enrichment.append(
                {
                    "weapon": w,
                    "weapon_class": get_weapon_class(w) or "unknown",
                    "high_count": high_count,
                    "low_count": low_count,
                    "high_pct": round(high_frac * 100, 2),
                    "low_pct": round(low_frac * 100, 2),
                    "enrichment_ratio": (
                        round(ratio, 2) if ratio != float("inf") else "∞"
                    ),
                }
            )

        # Sort by enrichment (high activation weapons first)
        enrichment.sort(
            key=lambda x: x["high_count"] - x["low_count"], reverse=True
        )

        result.add_table(
            "weapon_enrichment",
            enrichment[:30],
            columns=[
                "weapon",
                "weapon_class",
                "high_count",
                "low_count",
                "high_pct",
                "low_pct",
                "enrichment_ratio",
            ],
            description=f"Weapon distribution: top {high_pct}% vs bottom {low_pct}%",
        )

        # Sample high-activation examples with tokens
        sample_examples = []
        for ex in high_examples[:10]:
            weapon = ctx.id_to_weapon_display_name(ex["weapon_id"])
            sample_examples.append(
                {
                    "activation": round(ex["activation"], 4),
                    "weapon": weapon,
                    "tokens": ", ".join(ex["tokens"][:8]),  # First 8 tokens
                }
            )

        result.add_table(
            "high_activation_samples",
            sample_examples,
            columns=["activation", "weapon", "tokens"],
            description="Sample high-activation examples",
        )

        # Aggregates
        result.aggregates.n_samples = n
        result.aggregates.custom["high_group_size"] = high_n
        result.aggregates.custom["low_group_size"] = low_n

        if enrichment:
            top_enriched = max(
                [
                    e
                    for e in enrichment
                    if isinstance(e["enrichment_ratio"], (int, float))
                ],
                key=lambda x: x["enrichment_ratio"],
                default=None,
            )
            if top_enriched:
                result.aggregates.custom["most_enriched_weapon"] = top_enriched[
                    "weapon"
                ]

        result.diagnostics.n_contexts_tested = n
        logger.info(
            f"Compared {high_n} high vs {low_n} low activation examples"
        )

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
