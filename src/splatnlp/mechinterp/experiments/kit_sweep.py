"""Kit sweep experiment runner.

This module implements analysis of feature activation by weapon kit
(sub weapon and special weapon).
"""

import logging
import statistics
from collections import defaultdict

import polars as pl

from splatnlp.dashboard.utils.converters import (
    get_translation,
    get_weapon_properties,
)
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


def _get_kit_for_weapon_token(
    token_id: int, ctx: MechInterpContext
) -> tuple[str | None, str | None]:
    """Get (sub, special) for a weapon token ID.

    Args:
        token_id: The token ID from the model vocabulary
        ctx: MechInterp context with vocab mappings

    Returns:
        Tuple of (sub_name, special_name) or (None, None) if not found
    """
    # Find the weapon_id_XXXX key for this token ID
    inv_weapon_vocab = {v: k for k, v in ctx.weapon_vocab.items()}
    weapon_token = inv_weapon_vocab.get(token_id)

    if not weapon_token or not weapon_token.startswith("weapon_id_"):
        return None, None

    # Extract game ID (e.g., "weapon_id_1111" -> "1111")
    game_id = weapon_token.replace("weapon_id_", "")

    # Get weapon properties
    try:
        props = get_weapon_properties()
        weapon_props = props.get(game_id)
        if not weapon_props:
            return None, None

        # Get translations for sub/special
        translation = get_translation()
        subs = translation["USen"]["WeaponName_Sub"]
        specials = translation["USen"]["WeaponName_Special"]

        sub_key = weapon_props.get("sub", "")
        special_key = weapon_props.get("special", "")

        sub_name = subs.get(sub_key, sub_key)
        special_name = specials.get(special_key, special_key)

        return sub_name, special_name
    except Exception as e:
        logger.warning(f"Failed to get kit for weapon {game_id}: {e}")
        return None, None


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
class KitSweepRunner(ExperimentRunner):
    """Runner for kit (sub/special) sweep experiments.

    Analyzes how a feature responds to different sub weapons and
    special weapons, helping identify kit-linked patterns.
    """

    name = "kit_sweep"
    handles_types = [ExperimentType.KIT_SWEEP]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute kit sweep analysis."""
        vars_dict = spec.variables
        min_examples = vars_dict.get("min_examples", 10)
        top_k = vars_dict.get("top_k", 10)
        analyze_combinations = vars_dict.get("analyze_combinations", False)

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
        logger.info(f"Processing {len(data)} examples for kit analysis")

        # Group by sub and special
        sub_groups: dict[str, list[float]] = defaultdict(list)
        special_groups: dict[str, list[float]] = defaultdict(list)
        combo_groups: dict[tuple[str, str], list[float]] = defaultdict(list)

        for row in data:
            weapon_id = row.get("weapon_id", 0)
            activation = row.get("activation", 0)

            sub_name, special_name = _get_kit_for_weapon_token(weapon_id, ctx)

            if sub_name:
                sub_groups[sub_name].append(activation)
            if special_name:
                special_groups[special_name].append(activation)
            if sub_name and special_name and analyze_combinations:
                combo_groups[(sub_name, special_name)].append(activation)

        # Compute global mean
        all_acts = [row["activation"] for row in data]
        global_mean = statistics.mean(all_acts) if all_acts else 0

        # Compute sub stats
        sub_stats = []
        for sub_name, acts in sub_groups.items():
            if len(acts) < min_examples:
                continue
            mean_act = statistics.mean(acts)
            sub_stats.append(
                {
                    "sub": sub_name,
                    "mean_activation": round(mean_act, 4),
                    "std": round(
                        statistics.stdev(acts) if len(acts) > 1 else 0, 4
                    ),
                    "n": len(acts),
                    "delta_from_global": round(mean_act - global_mean, 4),
                }
            )

        sub_stats.sort(key=lambda x: x["mean_activation"], reverse=True)
        sub_stats = sub_stats[:top_k]

        result.add_table(
            "sub_stats",
            sub_stats,
            columns=[
                "sub",
                "mean_activation",
                "std",
                "n",
                "delta_from_global",
            ],
            description="Activation statistics by sub weapon",
        )

        # Compute special stats
        special_stats = []
        for special_name, acts in special_groups.items():
            if len(acts) < min_examples:
                continue
            mean_act = statistics.mean(acts)
            special_stats.append(
                {
                    "special": special_name,
                    "mean_activation": round(mean_act, 4),
                    "std": round(
                        statistics.stdev(acts) if len(acts) > 1 else 0, 4
                    ),
                    "n": len(acts),
                    "delta_from_global": round(mean_act - global_mean, 4),
                }
            )

        special_stats.sort(key=lambda x: x["mean_activation"], reverse=True)
        special_stats = special_stats[:top_k]

        result.add_table(
            "special_stats",
            special_stats,
            columns=[
                "special",
                "mean_activation",
                "std",
                "n",
                "delta_from_global",
            ],
            description="Activation statistics by special weapon",
        )

        # Compute combination stats if requested
        if analyze_combinations and combo_groups:
            combo_stats = []
            for (sub_name, special_name), acts in combo_groups.items():
                if len(acts) < min_examples:
                    continue
                mean_act = statistics.mean(acts)
                combo_stats.append(
                    {
                        "sub": sub_name,
                        "special": special_name,
                        "mean_activation": round(mean_act, 4),
                        "std": round(
                            statistics.stdev(acts) if len(acts) > 1 else 0, 4
                        ),
                        "n": len(acts),
                        "delta_from_global": round(mean_act - global_mean, 4),
                    }
                )

            combo_stats.sort(key=lambda x: x["mean_activation"], reverse=True)
            combo_stats = combo_stats[:top_k]

            result.add_table(
                "combo_stats",
                combo_stats,
                columns=[
                    "sub",
                    "special",
                    "mean_activation",
                    "std",
                    "n",
                    "delta_from_global",
                ],
                description="Activation statistics by sub+special combination",
            )

        # Populate aggregates
        if sub_stats:
            top_sub = sub_stats[0]
            result.aggregates.custom["top_sub"] = top_sub["sub"]
            result.aggregates.custom["top_sub_mean"] = top_sub[
                "mean_activation"
            ]
            result.aggregates.custom["top_sub_delta"] = top_sub[
                "delta_from_global"
            ]

        if special_stats:
            top_special = special_stats[0]
            result.aggregates.custom["top_special"] = top_special["special"]
            result.aggregates.custom["top_special_mean"] = top_special[
                "mean_activation"
            ]
            result.aggregates.custom["top_special_delta"] = top_special[
                "delta_from_global"
            ]

        result.aggregates.base_activation_mean = round(global_mean, 4)
        result.aggregates.n_samples = len(data)
        result.aggregates.n_conditions = len(sub_stats) + len(special_stats)
        result.diagnostics.n_contexts_tested = len(data)

        logger.info(
            f"Analyzed {len(sub_stats)} subs, {len(special_stats)} specials"
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
