"""Experiment specification schemas for mechanistic interpretability.

This module defines the data structures for specifying experiments
that can be executed by the mechinterp-runner skill.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ExperimentType(str, Enum):
    """Types of experiments that can be run."""

    # Family-based sweeps
    FAMILY_1D_SWEEP = "family_1d_sweep"
    FAMILY_2D_HEATMAP = "family_2d_heatmap"

    # Pattern mining
    FREQUENT_ITEMSETS = "frequent_itemsets"
    MINIMAL_CORES = "minimal_cores"

    # Interaction analysis
    PAIRWISE_INTERACTIONS = "pairwise_interactions"
    CONDITIONAL_INTERACTIONS = "conditional_interactions"
    WITHIN_FAMILY_INTERFERENCE = "within_family_interference"

    # Validation
    SPLIT_HALF = "split_half"
    SHUFFLE_NULL = "shuffle_null"

    # Weapon analysis
    WEAPON_SWEEP = "weapon_sweep"
    WEAPON_GROUP_ANALYSIS = "weapon_group_analysis"
    KIT_SWEEP = "kit_sweep"

    # Token influence analysis
    TOKEN_INFLUENCE_SWEEP = "token_influence_sweep"
    BINARY_PRESENCE_EFFECT = "binary_presence_effect"

    # Coverage and decoder analysis
    CORE_COVERAGE_ANALYSIS = "core_coverage_analysis"
    DECODER_OUTPUT_ANALYSIS = "decoder_output_analysis"


class DatasetSlice(BaseModel):
    """Configuration for slicing the dataset.

    Experiments operate on a subset of examples, typically filtered by
    activation percentile, weapon, or other criteria.
    """

    percentile_min: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum activation percentile to include",
    )
    percentile_max: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Maximum activation percentile to include",
    )
    activation_threshold: float | None = Field(
        default=None,
        description="Absolute activation threshold (alternative to percentiles)",
    )
    weapon_filter: list[int] | None = Field(
        default=None,
        description="List of weapon IDs to include (None = all weapons)",
    )
    weapon_exclude: list[int] | None = Field(
        default=None,
        description="List of weapon IDs to exclude",
    )
    family_filter: list[str] | None = Field(
        default=None,
        description="Only include examples with these ability families",
    )
    sample_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of examples to sample",
    )
    random_seed: int | None = Field(
        default=42,
        description="Random seed for reproducible sampling",
    )

    @field_validator("percentile_max")
    @classmethod
    def max_greater_than_min(cls, v: float, info) -> float:
        if "percentile_min" in info.data and v < info.data["percentile_min"]:
            raise ValueError("percentile_max must be >= percentile_min")
        return v


class FamilySweepVariables(BaseModel):
    """Variables for family sweep experiments."""

    family: str = Field(
        ..., description="Ability family to sweep (e.g., 'swim_speed_up')"
    )
    rungs: list[int] | None = Field(
        default=None,
        description="Specific AP rungs to test (None = all standard rungs)",
    )
    include_absent: bool = Field(
        default=True,
        description="Include baseline with family absent",
    )


class Family2DVariables(BaseModel):
    """Variables for 2D family heatmap experiments."""

    family_x: str = Field(..., description="First ability family (x-axis)")
    family_y: str = Field(..., description="Second ability family (y-axis)")
    rungs_x: list[int] | None = Field(
        default=None,
        description="AP rungs for x-axis family (None = all)",
    )
    rungs_y: list[int] | None = Field(
        default=None,
        description="AP rungs for y-axis family (None = all)",
    )
    base_weapon: int | None = Field(
        default=None,
        description="Fix weapon for consistent context",
    )


class ItemsetVariables(BaseModel):
    """Variables for frequent itemset mining."""

    min_support: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Minimum support threshold",
    )
    max_size: int = Field(
        default=4,
        ge=2,
        le=6,
        description="Maximum itemset size to mine",
    )
    collapse_families: bool = Field(
        default=False,
        description="Collapse AP rungs to family presence/absence",
    )
    condition_on: str | None = Field(
        default=None,
        description="Token to condition on (e.g., 'special_charge_up_57')",
    )
    high_activation_pct: float = Field(
        default=10.0,
        description="Top percentile to consider 'high activation'",
    )


class MinimalCoreVariables(BaseModel):
    """Variables for minimal activating core analysis."""

    retention_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Minimum activation fraction to retain",
    )
    clamp_families: list[str] = Field(
        default_factory=list,
        description="Families to keep in all cores (not removed during search)",
    )
    exclude_families: list[str] = Field(
        default_factory=list,
        description="Families to always remove from cores",
    )
    max_examples: int = Field(
        default=100,
        description="Maximum examples to analyze",
    )


class InteractionVariables(BaseModel):
    """Variables for interaction analysis."""

    candidate_tokens: list[str] | None = Field(
        default=None,
        description="Tokens to compute interactions for (None = top PageRank)",
    )
    n_candidates: int = Field(
        default=20,
        description="Number of top tokens if candidate_tokens is None",
    )
    family_mode: bool = Field(
        default=False,
        description="Collapse tokens to families before computing",
    )
    modulator_token: str | None = Field(
        default=None,
        description="For conditional: token that modulates interactions",
    )


class ValidationVariables(BaseModel):
    """Variables for validation experiments."""

    n_splits: int = Field(
        default=10,
        description="Number of random splits for split-half",
    )
    n_shuffles: int = Field(
        default=100,
        description="Number of shuffles for null distribution",
    )
    metric: str = Field(
        default="pagerank_correlation",
        description="Metric to validate stability of",
    )


class WeaponSweepVariables(BaseModel):
    """Variables for weapon sweep experiments."""

    condition_family: str | None = Field(
        default=None,
        description="Only consider examples with this family present",
    )
    min_examples: int = Field(
        default=10,
        description="Minimum examples per weapon to include",
    )
    top_k_weapons: int = Field(
        default=20,
        description="Limit output to top K weapons",
    )


class WeaponGroupVariables(BaseModel):
    """Variables for weapon group analysis."""

    high_percentile: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Top percentile for high activation group",
    )
    low_percentile: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Bottom percentile for low activation group",
    )


class KitSweepVariables(BaseModel):
    """Variables for kit (sub/special) sweep experiments."""

    min_examples: int = Field(
        default=10,
        ge=1,
        description="Minimum examples per kit to include in results",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of top subs/specials to report",
    )
    analyze_combinations: bool = Field(
        default=False,
        description="Also analyze sub+special combinations",
    )


class InterferenceVariables(BaseModel):
    """Variables for within-family interference analysis."""

    family: str | None = Field(
        default=None,
        description="Ability family to analyze (e.g., 'special_charge_up')",
    )
    rungs: list[int] | None = Field(
        default=None,
        description="Specific AP rungs to test (None = all standard rungs)",
    )


class TokenInfluenceVariables(BaseModel):
    """Variables for token influence sweep experiments."""

    min_samples: int = Field(
        default=50,
        ge=10,
        description="Minimum samples for token to be included",
    )
    high_percentile: float = Field(
        default=0.995,
        gt=0.5,
        le=1.0,
        description="Percentile threshold for 'high activation' (default top 0.5%)",
    )
    collapse_families: bool = Field(
        default=True,
        description="Collapse AP rungs to family names",
    )
    suppressor_threshold: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="High-rate ratio below which token is considered suppressor",
    )
    enhancer_threshold: float = Field(
        default=1.2,
        ge=1.0,
        description="High-rate ratio above which token is considered enhancer",
    )


class CoreCoverageVariables(BaseModel):
    """Variables for core coverage analysis.

    Checks if tokens are tail markers vs primary drivers by computing
    coverage in the core region (25-75% of effective max).
    """

    tokens_to_check: list[str] | None = Field(
        default=None,
        description="Specific tokens to check (None = use top PageRank tokens)",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of top PageRank tokens to check if tokens_to_check is None",
    )
    coverage_threshold: float = Field(
        default=0.30,
        gt=0.0,
        le=1.0,
        description="Below this coverage = tail marker (default 30%)",
    )
    include_weapons: bool = Field(
        default=True,
        description="Also compute weapon coverage (detect weapon flanderization)",
    )
    core_lower_pct: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="Lower bound of core region as fraction of effective max",
    )
    core_upper_pct: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Upper bound of core region as fraction of effective max",
    )
    effective_max_percentile: float = Field(
        default=0.995,
        gt=0.9,
        le=1.0,
        description="Percentile of nonzero activations to use as effective max",
    )


class DecoderOutputVariables(BaseModel):
    """Variables for decoder output analysis.

    Analyzes what tokens a feature PROMOTES/SUPPRESSES via the path:
    feature_decoder_vector → output_layer_weights → token logits
    """

    top_k_promoted: int = Field(
        default=15,
        ge=1,
        description="Number of top promoted tokens to return",
    )
    top_k_suppressed: int = Field(
        default=15,
        ge=1,
        description="Number of top suppressed tokens to return",
    )
    group_by_family: bool = Field(
        default=True,
        description="Aggregate results by ability family",
    )
    include_ap_level: bool = Field(
        default=True,
        description="Include AP level patterns (_3, _6, _57 etc.) in analysis",
    )
    exclude_special_tokens: bool = Field(
        default=True,
        description="Exclude <PAD>, <MASK>, <UNK> from results",
    )


class ExperimentSpec(BaseModel):
    """Complete specification for an experiment.

    This is the primary input to the mechinterp-runner. It contains all
    information needed to reproduce an experiment.
    """

    # Core identification
    id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        description="Unique experiment ID (auto-generated from timestamp)",
    )
    type: ExperimentType = Field(..., description="Type of experiment to run")
    feature_id: int = Field(..., description="SAE feature ID to analyze")
    model_type: Literal["full", "ultra"] = Field(
        ...,
        description="Model type (full=2K features, ultra=24K features)",
    )

    # Dataset configuration
    dataset_slice: DatasetSlice = Field(
        default_factory=DatasetSlice,
        description="Dataset slicing configuration",
    )

    # Experiment-specific variables
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Experiment-type-specific variables",
    )

    # Constraints
    constraints: list[str] = Field(
        default_factory=lambda: ["one_rung_per_family"],
        description="Domain constraint IDs to enforce",
    )

    # Output configuration
    outputs: dict[str, bool] = Field(
        default_factory=lambda: {
            "aggregates": True,
            "tables": True,
            "diagnostics": True,
            "figures": False,
        },
        description="Which outputs to generate",
    )

    # Metadata
    description: str | None = Field(
        default=None,
        description="Human-readable description of experiment purpose",
    )
    parent_hypothesis: str | None = Field(
        default=None,
        description="Hypothesis ID this experiment tests",
    )
    rationale: str | None = Field(
        default=None,
        description="Why this experiment was chosen",
    )
    created_at: datetime = Field(default_factory=datetime.now)

    def get_typed_variables(self) -> BaseModel | dict:
        """Get variables as the appropriate typed model."""
        type_map = {
            ExperimentType.FAMILY_1D_SWEEP: FamilySweepVariables,
            ExperimentType.FAMILY_2D_HEATMAP: Family2DVariables,
            ExperimentType.FREQUENT_ITEMSETS: ItemsetVariables,
            ExperimentType.MINIMAL_CORES: MinimalCoreVariables,
            ExperimentType.PAIRWISE_INTERACTIONS: InteractionVariables,
            ExperimentType.CONDITIONAL_INTERACTIONS: InteractionVariables,
            ExperimentType.WITHIN_FAMILY_INTERFERENCE: InterferenceVariables,
            ExperimentType.SPLIT_HALF: ValidationVariables,
            ExperimentType.SHUFFLE_NULL: ValidationVariables,
            ExperimentType.WEAPON_SWEEP: WeaponSweepVariables,
            ExperimentType.WEAPON_GROUP_ANALYSIS: WeaponGroupVariables,
            ExperimentType.KIT_SWEEP: KitSweepVariables,
            ExperimentType.TOKEN_INFLUENCE_SWEEP: TokenInfluenceVariables,
            ExperimentType.CORE_COVERAGE_ANALYSIS: CoreCoverageVariables,
            ExperimentType.DECODER_OUTPUT_ANALYSIS: DecoderOutputVariables,
        }
        model_class = type_map.get(self.type)
        if model_class:
            return model_class(**self.variables)
        return self.variables

    def to_filename(self) -> str:
        """Generate a descriptive filename for this spec."""
        type_short = self.type.value.replace("_", "-")
        return f"{self.id}__f{self.feature_id}__{type_short}.json"
