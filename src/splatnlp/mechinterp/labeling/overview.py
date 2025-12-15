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


# =============================================================================
# Extended Analysis Functions (for overview_cli --enrichment, --regions, etc.)
# =============================================================================


@dataclass
class TokenEnrichment:
    """Token enrichment statistics for high vs low activation."""

    token: str
    family: str
    ap_level: int | None
    baseline_rate: float  # Rate in all examples
    high_rate: float  # Rate in high-activation examples
    enrichment_ratio: float  # high_rate / baseline_rate
    is_enhancer: bool  # enrichment_ratio > 1.2
    is_suppressor: bool  # enrichment_ratio < 0.8
    n_total: int
    n_high_with_token: int


@dataclass
class RegionBreakdown:
    """Activation region statistics."""

    region_name: str
    pct_range: tuple[float, float]  # (low, high) as % of effective max
    n_examples: int
    pct_of_total: float
    top_weapons: list[tuple[str, int, float]]  # (weapon_name, count, pct)
    top_families: list[tuple[str, float]]  # (family_name, frequency)


@dataclass
class BinaryEnrichment:
    """Binary ability enrichment statistics."""

    ability: str
    baseline_presence: float  # Rate in all examples
    high_presence: float  # Rate in high-activation examples
    enrichment: float  # high_presence / baseline_presence
    mean_with: float
    mean_without: float
    delta: float
    n_with: int
    n_without: int


@dataclass
class KitBreakdown:
    """Sub/special weapon breakdown for a region."""

    region: str
    n_examples: int
    subs: list[tuple[str, int, float]]  # (sub_name, count, pct)
    specials: list[tuple[str, int, float]]  # (special_name, count, pct)


# Binary abilities (main-only, no AP scaling)
BINARY_ABILITIES = [
    "comeback",
    "stealth_jump",
    "last_ditch_effort",
    "ninja_squid",
    "respawn_punisher",
    "object_shredder",
    "drop_roller",
    "opening_gambit",
    "tenacity",
    "haunt",
]


def compute_token_enrichment(
    feature_id: int,
    ctx: MechInterpContext,
    high_percentile: float = 0.90,
    min_baseline_rate: float = 0.01,
) -> list[TokenEnrichment]:
    """Compute enrichment ratios for all tokens.

    Enrichment = (rate in high-activation examples) / (rate in all examples)
    - Ratio > 1.2 = enhancer (over-represented in high activation)
    - Ratio < 0.8 = suppressor (under-represented in high activation)

    Args:
        feature_id: SAE feature ID
        ctx: MechInterp context
        high_percentile: Top X% examples count as "high activation" (default: 10%)
        min_baseline_rate: Minimum baseline rate to include token (default: 1%)

    Returns:
        List of TokenEnrichment sorted by enrichment_ratio (suppressors first)
    """
    import polars as pl

    df = ctx.db.get_all_feature_activations_for_pagerank(feature_id)
    n_total = len(df)
    if n_total == 0:
        return []

    # Get high activation threshold
    threshold = df["activation"].quantile(high_percentile)
    high_df = df.filter(pl.col("activation") >= threshold)
    n_high = len(high_df)

    if n_high == 0:
        return []

    # Count token presence in all vs high
    all_tokens: dict[int, int] = defaultdict(int)
    high_tokens: dict[int, int] = defaultdict(int)

    for row in df.iter_rows(named=True):
        for tid in row.get("ability_input_tokens", []):
            if tid not in [0, 1, 2, 3]:  # Skip special tokens
                all_tokens[tid] += 1

    for row in high_df.iter_rows(named=True):
        for tid in row.get("ability_input_tokens", []):
            if tid not in [0, 1, 2, 3]:
                high_tokens[tid] += 1

    # Compute enrichment
    enrichments: list[TokenEnrichment] = []

    for tid, all_count in all_tokens.items():
        baseline_rate = all_count / n_total
        if baseline_rate < min_baseline_rate:
            continue

        high_count = high_tokens.get(tid, 0)
        high_rate = high_count / n_high

        enrichment_ratio = high_rate / baseline_rate if baseline_rate > 0 else 0

        token_name = ctx.inv_vocab.get(tid, f"token_{tid}")
        family, ap_level = parse_token(token_name)

        enrichments.append(
            TokenEnrichment(
                token=token_name,
                family=family or token_name,
                ap_level=ap_level,
                baseline_rate=round(baseline_rate, 4),
                high_rate=round(high_rate, 4),
                enrichment_ratio=round(enrichment_ratio, 4),
                is_enhancer=enrichment_ratio > 1.2,
                is_suppressor=enrichment_ratio < 0.8,
                n_total=all_count,
                n_high_with_token=high_count,
            )
        )

    # Sort by enrichment (suppressors first)
    enrichments.sort(key=lambda x: x.enrichment_ratio)

    return enrichments


def compute_region_breakdown(
    feature_id: int,
    ctx: MechInterpContext,
    effective_max_percentile: float = 99.5,
) -> list[RegionBreakdown]:
    """Compute activation breakdown by region for anti-flanderization analysis.

    Regions (as % of effective max activation):
    - Floor: ≤1%
    - Low: 1-10%
    - Below Core: 10-25%
    - Core: 25-75% (TRUE CONCEPT)
    - High: 75-90%
    - Flanderization: 90%+ (potential super-stimuli)

    Args:
        feature_id: SAE feature ID
        ctx: MechInterp context
        effective_max_percentile: Percentile for effective max (default: 99.5)

    Returns:
        List of RegionBreakdown for each region
    """
    import numpy as np
    import polars as pl

    df = ctx.db.get_all_feature_activations_for_pagerank(feature_id)
    n_total = len(df)
    if n_total == 0:
        return []

    acts = df["activation"].to_numpy()
    weapons = df["weapon_id"].to_list()
    token_lists = df["ability_input_tokens"].to_list()

    # Compute effective max (99.5th percentile of nonzero)
    nonzero_acts = acts[acts > 0]
    if len(nonzero_acts) == 0:
        return []

    effective_max = np.percentile(nonzero_acts, effective_max_percentile)

    # Define regions
    regions_def = [
        ("Floor (≤1%)", 0.0, 0.01),
        ("Low (1-10%)", 0.01, 0.10),
        ("Below Core (10-25%)", 0.10, 0.25),
        ("Core (25-75%)", 0.25, 0.75),
        ("High (75-90%)", 0.75, 0.90),
        ("Flanderization (90%+)", 0.90, 1.01),
    ]

    breakdowns: list[RegionBreakdown] = []

    for region_name, low_pct, high_pct in regions_def:
        low_thresh = low_pct * effective_max
        high_thresh = high_pct * effective_max

        # Find indices in this region
        if high_pct > 1.0:  # Flanderization: >=90%
            indices = [
                i for i, a in enumerate(acts) if a > low_thresh
            ]
        else:
            indices = [
                i
                for i, a in enumerate(acts)
                if low_thresh < a <= high_thresh
            ]

        n_region = len(indices)
        if n_region == 0:
            breakdowns.append(
                RegionBreakdown(
                    region_name=region_name,
                    pct_range=(low_pct * 100, high_pct * 100),
                    n_examples=0,
                    pct_of_total=0.0,
                    top_weapons=[],
                    top_families=[],
                )
            )
            continue

        # Count weapons in region
        weapon_counts: dict[int, int] = defaultdict(int)
        family_counts: dict[str, int] = defaultdict(int)

        for idx in indices:
            weapon_counts[weapons[idx]] += 1
            for tid in token_lists[idx]:
                tname = ctx.inv_vocab.get(tid, "")
                family, _ = parse_token(tname)
                if family:
                    family_counts[family] += 1

        # Top 5 weapons
        top_weapons = []
        for wid, count in sorted(
            weapon_counts.items(), key=lambda x: -x[1]
        )[:5]:
            wname = ctx.id_to_weapon_display_name(wid)
            pct = count / n_region * 100
            top_weapons.append((wname, count, round(pct, 1)))

        # Top 5 families
        top_families = []
        total_family = sum(family_counts.values())
        for fam, count in sorted(
            family_counts.items(), key=lambda x: -x[1]
        )[:5]:
            freq = count / total_family if total_family > 0 else 0
            top_families.append((fam, round(freq, 3)))

        breakdowns.append(
            RegionBreakdown(
                region_name=region_name,
                pct_range=(low_pct * 100, high_pct * 100),
                n_examples=n_region,
                pct_of_total=round(n_region / n_total * 100, 1),
                top_weapons=top_weapons,
                top_families=top_families,
            )
        )

    return breakdowns


def compute_binary_enrichment(
    feature_id: int,
    ctx: MechInterpContext,
    high_percentile: float = 0.90,
) -> list[BinaryEnrichment]:
    """Compute enrichment for binary abilities (main-only abilities).

    Binary abilities don't scale with AP, so 1D sweeps show delta=0.
    This analysis checks presence rate and mean activation instead.

    Args:
        feature_id: SAE feature ID
        ctx: MechInterp context
        high_percentile: Top X% examples count as "high activation"

    Returns:
        List of BinaryEnrichment sorted by enrichment (most depleted first)
    """
    import polars as pl

    df = ctx.db.get_all_feature_activations_for_pagerank(feature_id)
    n_total = len(df)
    if n_total == 0:
        return []

    # Get high activation threshold
    threshold = df["activation"].quantile(high_percentile)
    high_df = df.filter(pl.col("activation") >= threshold)
    n_high = len(high_df)

    if n_high == 0:
        return []

    enrichments: list[BinaryEnrichment] = []

    for ability in BINARY_ABILITIES:
        tok_id = ctx.vocab.get(ability)
        if tok_id is None:
            continue

        # Count presence
        with_binary_all = df.filter(
            pl.col("ability_input_tokens").list.contains(tok_id)
        )
        with_binary_high = high_df.filter(
            pl.col("ability_input_tokens").list.contains(tok_id)
        )

        n_with_all = len(with_binary_all)
        n_with_high = len(with_binary_high)

        if n_with_all == 0:
            continue

        # Compute rates
        baseline_presence = n_with_all / n_total
        high_presence = n_with_high / n_high
        enrichment = (
            high_presence / baseline_presence if baseline_presence > 0 else 0
        )

        # Compute mean activation WITH vs WITHOUT
        without_binary = df.filter(
            ~pl.col("ability_input_tokens").list.contains(tok_id)
        )

        mean_with = with_binary_all["activation"].mean()
        mean_without = without_binary["activation"].mean()
        delta = mean_with - mean_without

        enrichments.append(
            BinaryEnrichment(
                ability=ability,
                baseline_presence=round(baseline_presence, 4),
                high_presence=round(high_presence, 4),
                enrichment=round(enrichment, 2),
                mean_with=round(mean_with, 4),
                mean_without=round(mean_without, 4),
                delta=round(delta, 4),
                n_with=n_with_all,
                n_without=len(without_binary),
            )
        )

    # Sort by enrichment (most depleted first)
    enrichments.sort(key=lambda x: x.enrichment)

    return enrichments


def compute_kit_breakdown(
    feature_id: int,
    ctx: MechInterpContext,
    region: str = "core",
) -> KitBreakdown:
    """Compute sub/special weapon breakdown for a region.

    This reveals what sub weapons are actually being used when a feature
    like ISS (Ink Saver Sub) is enhanced.

    Args:
        feature_id: SAE feature ID
        ctx: MechInterp context
        region: Which region to analyze ("core", "high", "all")

    Returns:
        KitBreakdown with sub and special weapon counts
    """
    import numpy as np

    from splatnlp.dashboard.utils.converters import (
        get_translation,
        get_weapon_properties,
    )

    df = ctx.db.get_all_feature_activations_for_pagerank(feature_id)
    n_total = len(df)
    if n_total == 0:
        return KitBreakdown(region=region, n_examples=0, subs=[], specials=[])

    acts = df["activation"].to_numpy()
    weapons = df["weapon_id"].to_list()

    # Filter to region
    if region == "all":
        indices = list(range(n_total))
    else:
        nonzero_acts = acts[acts > 0]
        if len(nonzero_acts) == 0:
            return KitBreakdown(
                region=region, n_examples=0, subs=[], specials=[]
            )

        effective_max = np.percentile(nonzero_acts, 99.5)

        if region == "core":
            indices = [
                i
                for i, a in enumerate(acts)
                if 0.25 * effective_max < a <= 0.75 * effective_max
            ]
        elif region == "high":
            indices = [
                i for i, a in enumerate(acts) if a > 0.75 * effective_max
            ]
        else:
            indices = list(range(n_total))

    n_region = len(indices)
    if n_region == 0:
        return KitBreakdown(region=region, n_examples=0, subs=[], specials=[])

    # Load weapon properties and translations
    try:
        props = get_weapon_properties()
        translation = get_translation()
        subs_dict = translation.get("USen", {}).get("WeaponName_Sub", {})
        specials_dict = translation.get("USen", {}).get(
            "WeaponName_Special", {}
        )
    except Exception as e:
        logger.warning(f"Could not load weapon kit data: {e}")
        return KitBreakdown(region=region, n_examples=n_region, subs=[], specials=[])

    # Count subs and specials
    sub_counts: dict[str, int] = defaultdict(int)
    special_counts: dict[str, int] = defaultdict(int)

    for idx in indices:
        wid = weapons[idx]
        # Convert mechinterp weapon token ID to game weapon ID
        internal_name = ctx.inv_weapon_vocab.get(wid, "")
        if not internal_name.startswith("weapon_id_"):
            continue
        game_id = internal_name.replace("weapon_id_", "")

        weapon_props = props.get(game_id)
        if not weapon_props:
            continue

        # Get sub/special display names
        sub_code = weapon_props.get("sub", "")
        special_code = weapon_props.get("special", "")

        sub_name = subs_dict.get(sub_code, sub_code)
        special_name = specials_dict.get(special_code, special_code)

        if sub_name:
            sub_counts[sub_name] += 1
        if special_name:
            special_counts[special_name] += 1

    # Convert to sorted lists
    subs = [
        (name, count, round(count / n_region * 100, 1))
        for name, count in sorted(sub_counts.items(), key=lambda x: -x[1])
    ]
    specials = [
        (name, count, round(count / n_region * 100, 1))
        for name, count in sorted(special_counts.items(), key=lambda x: -x[1])
    ]

    return KitBreakdown(
        region=region, n_examples=n_region, subs=subs[:15], specials=specials[:15]
    )
