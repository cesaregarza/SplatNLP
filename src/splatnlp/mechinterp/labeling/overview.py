"""Feature overview for quick "first look" analysis.

This module provides a fast way to get an overview of a feature's
characteristics before deep investigation.
"""

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from splatnlp.dashboard.utils.pagerank import PageRankAnalyzer
from splatnlp.mechinterp.schemas.glossary import TOKEN_FAMILIES, parse_token
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


@dataclass
class SampleContext:
    """A single high-activation example context."""

    tokens: list[str]
    weapon: str
    activation: float


@dataclass
class TokenInfluence:
    """Statistics for a token's influence on feature activation."""

    token: str
    delta: float  # mean_present - mean_absent
    effect_size: float  # Cohen's d
    mean_present: float
    mean_absent: float
    n_present: int
    n_absent: int
    # High-activation enrichment: ratio of (token in high / token in low)
    # Values < 1 mean token is suppressed in high-activation examples
    high_rate_ratio: float = 1.0
    # Proportion of high-activation examples containing this token
    high_rate: float = 0.0
    # Proportion of low-activation examples containing this token
    low_rate: float = 0.0


@dataclass
class FeatureOverview:
    """Comprehensive first-look overview of a feature."""

    feature_id: int
    model_type: str

    # Activation stats
    activation_mean: float
    activation_std: float
    activation_median: float
    sparsity: float  # Percentage of zeros (0-100)
    n_examples: int

    # Top tokens by PageRank
    top_tokens: list[tuple[str, float]] = field(default_factory=list)

    # Family breakdown (aggregated from top tokens)
    family_breakdown: dict[str, float] = field(default_factory=dict)

    # Weapon breakdown
    top_weapons: list[tuple[str, int]] = field(default_factory=list)

    # Sample high-activation contexts
    sample_contexts: list[SampleContext] = field(default_factory=list)

    # Quick diagnostic flags
    relu_floor_rate: float = 0.0
    existing_label: str | None = None

    # Conceptual hints inferred from token usage
    concept_hints: list[str] = field(default_factory=list)

    # Bottom tokens (suppressors) - tokens with most negative influence
    bottom_tokens: list[tuple[str, float]] = field(default_factory=list)

    # Full token influence data for detailed analysis
    token_influences: list[TokenInfluence] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Format overview as markdown for display."""
        lines = [
            f"## Feature {self.feature_id} Overview ({self.model_type})",
            "",
        ]

        # Existing label
        if self.existing_label:
            lines.append(f"**Current Label:** {self.existing_label}")
            lines.append("")

        # Stats section
        lines.extend(
            [
                "### Activation Stats",
                f"- Mean: {self.activation_mean:.4f}",
                f"- Std: {self.activation_std:.4f}",
                f"- Median: {self.activation_median:.4f}",
                f"- Sparsity: {self.sparsity:.1f}%",
                f"- Examples: {self.n_examples:,}",
                "",
            ]
        )

        # ReLU floor warning
        if self.relu_floor_rate > 0.5:
            lines.append(
                f"**Warning:** ReLU floor detected ({self.relu_floor_rate:.1%} of contexts)"
            )
            lines.append("")

        # Top tokens
        if self.top_tokens:
            lines.append("### Top Tokens (PageRank)")
            for i, (token, score) in enumerate(self.top_tokens[:10], 1):
                lines.append(f"{i}. `{token}` ({score:.3f})")
            lines.append("")

        # Bottom tokens (suppressors)
        if self.bottom_tokens:
            lines.append("### Bottom Tokens (Suppressors)")
            lines.append(
                "*Tokens that rarely appear in high-activation examples "
                "(ratio < 1 = suppressed)*"
            )
            for i, (token, ratio) in enumerate(self.bottom_tokens[:10], 1):
                # Show as percentage: 0.5 ratio = "appears 50% as often"
                pct = ratio * 100
                lines.append(f"{i}. `{token}` ({pct:.1f}% of expected rate)")
            lines.append("")

        # Family breakdown
        if self.family_breakdown:
            lines.append("### Family Breakdown")
            sorted_families = sorted(
                self.family_breakdown.items(), key=lambda x: -x[1]
            )
            for family, score in sorted_families[:5]:
                pct = score * 100
                lines.append(f"- {family}: {pct:.1f}%")
            lines.append("")

        # Top weapons
        if self.top_weapons:
            lines.append("### Top Weapons")
            for weapon, count in self.top_weapons[:5]:
                lines.append(f"- {weapon}: {count:,}")
            lines.append("")

        # Sample contexts
        if self.sample_contexts:
            lines.append("### Sample Contexts (High Activation)")
            for i, ctx in enumerate(self.sample_contexts[:3], 1):
                tokens_str = ", ".join(ctx.tokens[:6])
                if len(ctx.tokens) > 6:
                    tokens_str += "..."
                lines.append(
                    f"{i}. [{ctx.weapon}] {tokens_str} (act={ctx.activation:.3f})"
                )
            lines.append("")

        if self.concept_hints:
            lines.append("### Concept Hints")
            for hint in self.concept_hints:
                lines.append(f"- {hint}")
            lines.append("")

        return "\n".join(lines)


def compute_overview(
    feature_id: int,
    ctx: MechInterpContext,
    top_k_tokens: int = 15,
    n_sample_contexts: int = 5,
    max_examples: int | None = 5000,
) -> FeatureOverview:
    """Compute a comprehensive overview for a feature.

    Args:
        feature_id: SAE feature ID
        ctx: MechInterp context with database and vocab
        top_k_tokens: Number of top tokens to include
        n_sample_contexts: Number of sample contexts to include
        max_examples: Maximum number of examples to load for quick analysis.
            If None, uses all available activations (slower).

    Returns:
        FeatureOverview with all computed fields
    """
    logger.info(f"Computing overview for feature {feature_id}")

    # Get basic stats (fast - precomputed)
    stats = ctx.db.get_feature_stats(feature_id)
    logger.debug(f"Got stats: {stats}")

    # Get feature summary for weapon breakdown
    summary = ctx.db.get_feature_summary(feature_id)

    # Get activations for PageRank and samples
    logger.info("Loading activation data for PageRank...")
    df: Any | None = None

    # When max_examples is None, we need ALL data for accurate token influence
    # computation. Use the full sweep method directly.
    if max_examples is None:
        try:
        df = ctx.db.get_all_feature_activations_for_pagerank(
            feature_id, limit=max_examples if max_examples else None
        )
            logger.info(
                f"Loaded {len(df) if df is not None else 0} rows via full sweep"
            )
        except Exception as e:
            logger.warning(f"Full sweep failed: {e}")
    else:
        # Fast path with limited examples
        fast_limit = max_examples
        if hasattr(ctx.db, "feature_lookup"):
            lookup = getattr(ctx.db, "feature_lookup", {})
            if feature_id in lookup:
                start_idx, end_idx = lookup[feature_id]
                available = max(0, end_idx - start_idx)
                if available > 0:
                    fast_limit = min(available, max_examples)

        try:
            df = ctx.db.get_feature_activations(feature_id, limit=fast_limit)
        except Exception as e:
            logger.debug(f"Fast activation load failed: {e}")

    # Fallback if we still don't have data
    if df is None or len(df) == 0:
        try:
            df = ctx.db.get_all_feature_activations_for_pagerank(feature_id)
        except AttributeError:
            try:
                df = ctx.db.get_feature_activations(
                    feature_id, limit=max_examples or 1000
                )
            except Exception as e:
                logger.warning(f"Error loading activations: {e}")
                df = None

    if df is None or len(df) == 0:
        logger.warning(f"No activation data for feature {feature_id}")
        return FeatureOverview(
            feature_id=feature_id,
            model_type=ctx.model_type,
            activation_mean=0,
            activation_std=0,
            activation_median=0,
            sparsity=100,
            n_examples=0,
        )

    # Collect lazy frames if needed and cap rows for speed
    if hasattr(df, "collect"):
        try:
            df = df.collect()
        except Exception as e:
            logger.debug(f"Collect failed, continuing with existing frame: {e}")

    if max_examples is not None and len(df) > max_examples:
        try:
            df = df.head(max_examples)
        except Exception as e:
            logger.debug(f"Head failed on activation frame: {e}")

    n_examples = len(df)
    logger.info(f"Loaded {n_examples} examples")

    # Compute PageRank for top tokens
    logger.info("Computing PageRank...")
    pr = PageRankAnalyzer(ctx.vocab, ctx.inv_vocab, mode="family")
    freq_counts: dict[str, int] = {}
    presence_counts: dict[str, int] = {}

    # NEW: Track activations per token for influence computation
    token_present_activations: dict[str, list[float]] = defaultdict(list)
    all_activations: list[float] = []

    for row in df.iter_rows(named=True):
        token_ids = row.get("ability_input_tokens", [])
        activation = row.get("activation", 0)
        pr.add_example(token_ids, activation)
        all_activations.append(activation)
        bases_seen = set()
        for tid in token_ids or []:
            tname = ctx.inv_vocab.get(tid, "")
            base = _token_base(tname)
            if base:
                freq_counts[base] = freq_counts.get(base, 0) + 1
                bases_seen.add(base)
        for base in bases_seen:
            presence_counts[base] = presence_counts.get(base, 0) + 1
            token_present_activations[base].append(activation)

    scores = pr.compute_pagerank()
    top_tokens_raw = pr.get_top_tokens(scores, top_k=top_k_tokens)

    # Convert to list of (name, score) tuples
    top_tokens = [(name, float(score)) for name, _, score in top_tokens_raw]

    # Aggregate by family
    family_scores: dict[str, float] = defaultdict(float)
    total_score = sum(score for _, score in top_tokens)

    for token_name, score in top_tokens:
        family, _ = parse_token(token_name)
        if family:
            family_scores[family] += score

    # Normalize to percentages
    if total_score > 0:
        family_breakdown = {
            fam: score / total_score for fam, score in family_scores.items()
        }
    else:
        family_breakdown = {}

    # Process weapon breakdown from summary
    top_weapons = []
    if "top_weapons" in summary:
        for wp in summary["top_weapons"]:
            weapon_id = wp.get("weapon_id", 0)
            count = wp.get("count", 0)
            weapon_name = ctx.id_to_weapon_display_name(weapon_id)
            top_weapons.append((weapon_name, count))

    concept_hints = _infer_concepts(
        df=df,
        ctx=ctx,
        top_tokens=top_tokens,
        freq_tokens=freq_counts,
        presence_counts=presence_counts,
    )

    # Compute token influence (delta and effect size)
    logger.info("Computing token influence...")
    token_influences = _compute_token_influences(
        token_present_activations=token_present_activations,
        all_activations=all_activations,
        min_samples=50,
    )

    # Extract bottom tokens (suppressed in high-activation examples)
    # Use high_rate_ratio < 0.8 as threshold (token appears <80% as often in high vs low)
    bottom_tokens = [
        (ti.token, ti.high_rate_ratio)
        for ti in token_influences
        if ti.high_rate_ratio < 0.8
    ][:10]

    # Get sample contexts
    sample_contexts = []
    df_sorted = df.sort("activation", descending=True)

    for row in df_sorted.head(n_sample_contexts).iter_rows(named=True):
        token_ids = row.get("ability_input_tokens", [])
        weapon_id = row.get("weapon_id", 0)
        activation = row.get("activation", 0)

        tokens = [ctx.inv_vocab.get(tid, "") for tid in token_ids]
        tokens = [t for t in tokens if t and not t.startswith("<")]
        weapon = ctx.id_to_weapon_display_name(weapon_id)

        sample_contexts.append(
            SampleContext(tokens=tokens, weapon=weapon, activation=activation)
        )

    # Compute ReLU floor rate
    zero_count = len(df.filter(df["activation"] <= 0.001))
    relu_floor_rate = zero_count / n_examples if n_examples > 0 else 0

    # Check for existing label
    existing_label = None
    try:
        from splatnlp.dashboard.components.feature_labels import (
            FeatureLabelsManager,
        )

        labels_manager = FeatureLabelsManager(ctx.model_type)
        label_data = labels_manager.get_label(feature_id)
        if label_data and label_data.get("name"):
            existing_label = label_data["name"]
    except Exception as e:
        logger.debug(f"Could not load existing label: {e}")

    return FeatureOverview(
        feature_id=feature_id,
        model_type=ctx.model_type,
        activation_mean=stats.get("mean", 0),
        activation_std=stats.get("std", 0),
        activation_median=stats.get("median", 0),
        sparsity=stats.get("sparsity", 0) * 100,
        n_examples=n_examples,
        top_tokens=top_tokens,
        family_breakdown=family_breakdown,
        top_weapons=top_weapons,
        sample_contexts=sample_contexts,
        relu_floor_rate=relu_floor_rate,
        existing_label=existing_label,
        concept_hints=concept_hints,
        bottom_tokens=bottom_tokens,
        token_influences=token_influences,
    )


def _infer_concepts(
    df: Any,
    ctx: MechInterpContext,
    top_tokens: list[tuple[str, float]],
    freq_tokens: dict[str, int],
    presence_counts: dict[str, int],
) -> list[str]:
    """Generate high-level concept hints from token statistics."""
    hints: list[str] = []
    n_examples = len(df) if df is not None else 0
    if n_examples == 0:
        return hints

    def coverage(token_bases: set[str]) -> float:
        total = sum(presence_counts.get(t, 0) for t in token_bases)
        return total / n_examples if n_examples else 0.0

    def has_top_token(token_base: str) -> bool:
        return any(tok.startswith(token_base) for tok, _ in top_tokens)

    # Core concept buckets
    death_tokens = {
        "quick_respawn",
        "respawn_punisher",
        "special_saver",
        "stealth_jump",
        "comeback",
    }
    opener_tokens = {"opening_gambit"}
    special_charge_tokens = {"special_charge_up"}
    mobility_tokens = {"swim_speed_up", "run_speed_up"}

    # Special Charge + opener stack
    sc_cov = min(coverage(special_charge_tokens), 1.0)
    og_cov = coverage(opener_tokens)
    if sc_cov > 0.15:
        msg = f"Strong SCU reliance (present in ~{sc_cov*100:.0f}% of sampled contexts)"
        if og_cov > 0.05:
            msg += " often paired with Opening Gambit"
        hints.append(msg)

    # Mobility
    mob_cov = coverage(mobility_tokens)
    if mob_cov > 0.1:
        hints.append(
            f"Mobility boost present (swim/run speed in ~{mob_cov*100:.0f}% of contexts)"
        )

    # Death-related signals: tolerate Comeback, avoid other death perks
    death_cov = min(coverage(death_tokens), 1.0)
    comeback_cov = min(coverage({"comeback"}), 1.0)
    non_cb_cov = death_cov - comeback_cov
    if death_cov > 0:
        if (
            comeback_cov > 0.02
            and non_cb_cov < 0.02
            and has_top_token("special_charge_up")
        ):
            hints.append(
                "Death-linked perks largely absent except Comeback, which co-occurs with SCU (death aversion with SCU-only exception)"
            )
        elif non_cb_cov > 0.05:
            hints.append(
                "Broad death-linked perks appear; consider testing respawn/penalty interactions explicitly"
            )
    else:
        # Fall back to top-token presence to catch rare samples
        if has_top_token("comeback") and not any(
            has_top_token(t) for t in ["quick_respawn", "respawn_punisher"]
        ):
            hints.append(
                "Comeback appears without other death perks—likely tolerates the SCU portion, avoids death-focus overall"
            )

    # Object play / situational cues
    if has_top_token("object_shredder"):
        hints.append(
            "Object Shredder shows up—feature may care about objective damage windows"
        )

    return hints


def _token_base(token: str) -> str:
    """Strip AP suffixes like _57 to get the ability family name."""
    if not token:
        return ""
    parts = token.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    return token


def _compute_token_influences(
    token_present_activations: dict[str, list[float]],
    all_activations: list[float],
    min_samples: int = 50,
    high_percentile: float = 0.995,
) -> list[TokenInfluence]:
    """Compute influence delta and effect size for each token.

    Args:
        token_present_activations: Dict mapping token -> list of activations when present
        all_activations: List of all activations (for computing absent stats)
        min_samples: Minimum samples for both present and absent to include token
        high_percentile: Percentile threshold for "high activation" (default top 0.5%)

    Returns:
        List of TokenInfluence sorted by high_rate_ratio (most suppressed first)
    """
    n_total = len(all_activations)
    if n_total == 0:
        return []

    # Compute high activation threshold (top 0.5% by default)
    sorted_acts = sorted(all_activations, reverse=True)
    high_idx = max(1, int(n_total * (1 - high_percentile)))
    high_threshold = sorted_acts[min(high_idx, len(sorted_acts) - 1)]

    n_high = sum(1 for a in all_activations if a >= high_threshold)
    n_low = n_total - n_high

    logger.debug(
        f"High activation threshold: {high_threshold:.4f} "
        f"(n_high={n_high}, n_low={n_low})"
    )

    sum_all = sum(all_activations)
    influences: list[TokenInfluence] = []

    for token, present_acts in token_present_activations.items():
        n_present = len(present_acts)
        n_absent = n_total - n_present

        # Skip tokens with insufficient samples
        if n_present < min_samples or n_absent < min_samples:
            continue

        # Count high/low activation examples containing this token
        n_high_with_token = sum(1 for a in present_acts if a >= high_threshold)
        n_low_with_token = n_present - n_high_with_token

        # Compute rates
        high_rate = n_high_with_token / n_high if n_high > 0 else 0
        low_rate = n_low_with_token / n_low if n_low > 0 else 0

        # Compute high_rate_ratio (with smoothing to avoid division by zero)
        # Ratio < 1 means token is suppressed in high-activation examples
        if low_rate > 0.001:
            high_rate_ratio = high_rate / low_rate
        elif high_rate > 0:
            high_rate_ratio = (
                10.0  # Token only appears in high (strong enhancer)
            )
        else:
            high_rate_ratio = 1.0  # Neither appears

        # Compute present stats
        mean_present = statistics.mean(present_acts)
        var_present = statistics.variance(present_acts) if n_present > 1 else 0

        # Compute absent mean from complement
        sum_present = sum(present_acts)
        mean_absent = (sum_all - sum_present) / n_absent if n_absent > 0 else 0

        # Compute absent variance
        sum_sq_all = sum(a * a for a in all_activations)
        sum_sq_present = sum(a * a for a in present_acts)
        sum_sq_absent = sum_sq_all - sum_sq_present

        if n_absent > 1:
            mean_sq_absent = sum_sq_absent / n_absent
            var_absent = max(0, mean_sq_absent - mean_absent * mean_absent)
        else:
            var_absent = 0

        # Compute delta
        delta = mean_present - mean_absent

        # Compute Cohen's d effect size
        pooled_var = (var_present * n_present + var_absent * n_absent) / (
            n_present + n_absent
        )
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
        effect_size = delta / pooled_std

        influences.append(
            TokenInfluence(
                token=token,
                delta=round(delta, 6),
                effect_size=round(effect_size, 4),
                mean_present=round(mean_present, 6),
                mean_absent=round(mean_absent, 6),
                n_present=n_present,
                n_absent=n_absent,
                high_rate_ratio=round(high_rate_ratio, 4),
                high_rate=round(high_rate, 4),
                low_rate=round(low_rate, 4),
            )
        )

    # Sort by high_rate_ratio (most suppressed first = lowest ratio first)
    influences.sort(key=lambda x: x.high_rate_ratio)

    return influences
