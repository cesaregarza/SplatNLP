"""
Feature Intervals Grid Dashboard Component

A refactored version of the intervals grid visualization for feature analysis.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import polars as pl
from dash import ALL, Input, Output, State, callback, dcc, html

from splatnlp.dashboard.components.feature_labels import FeatureLabelsManager
from splatnlp.dashboard.fs_database import FSDatabase
from splatnlp.dashboard.utils.converters import (
    AbilityTagParser,
    generate_weapon_name_mapping,
    get_weapon_properties_df,
)
from splatnlp.dashboard.utils.tfidf import compute_tf_idf
from splatnlp.preprocessing.transform.mappings import generate_maps

logger = logging.getLogger(__name__)

# Constants
MAX_TFIDF_FEATURES = 10
MAX_SAMPLES_PER_BIN = 20  # Increased from 14 to show more examples
TOP_WEAPONS_COUNT = 5
TOP_BINS_FOR_ANALYSIS = 8
CARD_WIDTH = "250px"
CARD_HEIGHT = "220px"


@dataclass
class TFIDFAnalysis:
    """Results of TF-IDF analysis on feature activations."""

    top_tokens: list[dict[str, Any]]
    top_weapons: list[dict[str, Any]]
    sub_stats: list[dict[str, Any]]
    special_stats: list[dict[str, Any]]
    class_stats: list[dict[str, Any]]
    feature_display_name: str
    top_tfidf_token_names: set[str]  # Cache for UI highlighting


@dataclass
class FeatureAnalysisCache:
    """Cache for feature analysis to avoid recomputation.

    This cache stores all computed data for a single feature, including:
    - Raw activation and histogram data
    - TF-IDF analysis results
    - Weapon ID to name mappings

    The cache is invalidated when a different feature is selected.
    """

    feature_id: int
    activations_df: pl.DataFrame
    histogram_df: pl.DataFrame
    tfidf_analysis: Optional[TFIDFAnalysis] = None
    weapon_id_to_name: dict[int, str] = field(default_factory=dict)


@dataclass
class PercentileRangeData:
    """Data for a single percentile range of a feature.

    Used for slider-scale hypothesis testing: comparing how token/weapon
    distributions differ across activation value ranges.
    """

    feature_id: int
    feature_name: str
    percentile_low: float  # e.g., 40.0
    percentile_high: float  # e.g., 60.0
    activation_min: float
    activation_max: float
    example_count: int
    # Token frequencies: {token_name: (count, percentage)}
    token_distribution: dict[str, tuple[int, float]]
    # Weapon frequencies: {weapon_name: (count, percentage)}
    weapon_distribution: dict[str, tuple[int, float]]
    # Weapon properties: {weapon_name: {sub, special, class}}
    weapon_properties: dict[str, dict[str, str]]
    median_activation: float
    net_influences: list[dict]  # [{token, raw, net, direction}, ...]
    sample_examples: list[dict]  # [{weapon, abilities, activation}, ...]


class TFIDFAnalyzer:
    """Performs TF-IDF analysis on feature activations."""

    def __init__(
        self,
        inv_vocab: dict[str, str],
        inv_weapon_vocab: dict[int, str],
        id_to_name: dict[str, str],
        idf: pl.DataFrame,
    ):
        self.inv_vocab = inv_vocab
        self.inv_weapon_vocab = inv_weapon_vocab
        self.id_to_name = id_to_name
        self.parser = AbilityTagParser()
        self.idf = idf
        self.id_to_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
        self.weapon_properties = get_weapon_properties_df()

    def analyze(
        self,
        activations_df: pl.DataFrame,
        feature_labels_manager: FeatureLabelsManager,
        selected_feature_id: int,
    ) -> TFIDFAnalysis | None:
        """Perform TF-IDF analysis on the top activations."""
        # If IDF is empty, compute it from the current activations
        if self.idf.is_empty() or len(self.idf) == 0:
            from splatnlp.dashboard.utils.tfidf import compute_idf

            self.idf = compute_idf(activations_df)

        # Compute TF-IDF once
        tf_idf = compute_tf_idf(self.idf, activations_df)

        # Get top tokens with their scores
        top_tokens = (
            tf_idf.sort("tf_idf", descending=True)
            .head(MAX_TFIDF_FEATURES)
            .with_columns(
                pl.col("ability_input_tokens")
                .cast(pl.Utf8)
                .replace(self.inv_vocab)
                .alias("ability_name")
            )
            .select(["ability_name", "tf_idf"])
            .to_dicts()
        )

        # Extract token names for UI highlighting
        top_tfidf_token_names = {token["ability_name"] for token in top_tokens}

        # Compute weapon frequencies
        all_stats = self._compute_weapon_frequencies(
            activations_df, self.inv_weapon_vocab
        )

        # Get feature display name
        feature_display_name = self._get_feature_display_name(
            feature_labels_manager, selected_feature_id
        )

        return TFIDFAnalysis(
            top_tokens=top_tokens,
            top_weapons=all_stats["weapon_stats"],
            sub_stats=all_stats["sub_stats"],
            special_stats=all_stats["special_stats"],
            class_stats=all_stats["class_stats"],
            feature_display_name=feature_display_name,
            top_tfidf_token_names=top_tfidf_token_names,
        )

    def _compute_weapon_frequencies(
        self, df: pl.DataFrame, inv_weapon_vocab: dict[int, str]
    ) -> dict[str, list[dict[str, str | float]]]:
        """Calculate weapon frequencies in the dataframe.

        Uses COUNT of examples (not sum of activations) to rank weapons,
        so that weapons appearing most frequently in top activations rank highest.
        """
        total_examples = len(df)

        df = df.with_columns(
            pl.col("weapon_id_token")
            .cast(pl.Utf8)
            .replace(inv_weapon_vocab)
            .alias("weapon_id")
        ).join(
            self.weapon_properties,
            on="weapon_id",
            how="left",
        )

        # Group by weapon and COUNT examples (not sum activations)
        weapon_stats = (
            df.group_by("weapon_id_token")
            .agg(pl.len().alias("count"))
            .with_columns(
                [
                    pl.col("weapon_id_token")
                    .cast(pl.Utf8)
                    .replace(self.id_to_name_mapping)
                    .alias("weapon_name"),
                    (pl.col("count") / total_examples).alias("percentage"),
                ]
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["weapon_name", "percentage"])
            .to_dicts()
        )

        sub_stats = (
            df.group_by("sub")
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / total_examples).alias("percentage"),
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["sub", "percentage"])
            .to_dicts()
        )

        special_stats = (
            df.group_by("special")
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / total_examples).alias("percentage"),
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["special", "percentage"])
            .to_dicts()
        )

        class_stats = (
            df.group_by("class")
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / total_examples).alias("percentage"),
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["class", "percentage"])
            .to_dicts()
        )

        return {
            "weapon_stats": weapon_stats,
            "sub_stats": sub_stats,
            "special_stats": special_stats,
            "class_stats": class_stats,
        }

    def _get_feature_display_name(
        self,
        feature_labels_manager: FeatureLabelsManager | None,
        feature_id: int,
    ) -> str:
        """Get display name for feature."""
        if feature_labels_manager:
            return feature_labels_manager.get_display_name(feature_id)
        return f"Feature {feature_id}"


class UIComponentBuilder:
    """Builds UI components for the dashboard."""

    @staticmethod
    def build_tfidf_badge_color(score: float, all_scores: list[float]) -> str:
        """Determine badge color based on TF-IDF score percentile."""
        if not all_scores:
            return "primary"

        min_score = min(all_scores)
        max_score = max(all_scores)

        if max_score == min_score:
            return "primary"

        # Normalize score to 0-1 range
        normalized = (score - min_score) / (max_score - min_score)

        # Color gradient based on percentile
        if normalized > 0.8:
            return "danger"
        elif normalized > 0.6:
            return "warning"
        elif normalized > 0.4:
            return "primary"
        elif normalized > 0.2:
            return "info"
        else:
            return "secondary"

    @staticmethod
    def build_example_card(
        record: dict[str, Any],
        weapon_name: str,
        activation_val: float,
        top_tfidf_tokens: set[str],
        inv_vocab: dict[str, str],
    ) -> dbc.Card:
        """Build a card displaying an example activation."""
        # Generate a unique ID for the button's index
        # Using weapon_id_token and a formatted activation value
        weapon_id_token_str = str(record.get("weapon_id_token", "unknown"))
        activation_str = f"{record.get('activation', 0.0):.4f}"  # Format to avoid overly long/problematic strings
        unique_example_id = f"{weapon_id_token_str}_{activation_str}"

        # Prepare data for the data-attribute
        example_data_to_store = {
            "ability_input_tokens": record.get("ability_input_tokens"),
            "weapon_id_token": record.get("weapon_id_token"),
            "activation": record.get("activation"),
        }
        example_data_json = json.dumps(example_data_to_store)

        # Get ability names
        ability_names = [
            inv_vocab[token]
            for token in record["ability_input_tokens"]
            if token in inv_vocab
        ]

        # Format ability names with highlighting
        formatted_names = []
        for name in ability_names:
            if name in top_tfidf_tokens:
                formatted_names.append(html.Strong(name))
            else:
                formatted_names.append(name)

        ability_display = (
            html.Span(
                [html.Span(name, className="me-1") for name in formatted_names]
            )
            if formatted_names
            else "N/A"
        )

        # Build card
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H6(
                        weapon_name,
                        className="card-title mb-2 text-truncate",
                        title=weapon_name,
                        style={
                            "minHeight": "1.5rem"
                        },  # Ensure consistent height for title
                    ),
                    html.Div(
                        ability_display,
                        className="flex-grow-1 overflow-auto mb-2",
                        style={
                            "fontSize": "0.8rem",
                            "maxHeight": "calc(100% - 5rem)",
                        },  # Reserve space for title, activation and button
                    ),
                    html.P(
                        f"Activation: {activation_val:.4f}",
                        className="mb-0 fw-semibold text-primary",
                        style={
                            "minHeight": "1.5rem"
                        },  # Ensure consistent height for activation
                    ),
                    html.Div(
                        id={
                            "type": "ablation-card-data-wrapper",
                            "index": unique_example_id,
                        },
                        **{
                            "data-example": example_data_json
                        },  # Store data in wrapper
                        children=[
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Ablation",
                                        id={
                                            "type": "select-ablation-primary",
                                            "index": unique_example_id,
                                        },
                                        color="primary",
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        "+ Sweep",
                                        id={
                                            "type": "add-to-sweep-queue",
                                            "index": unique_example_id,
                                        },
                                        color="secondary",
                                        size="sm",
                                    ),
                                ],
                                size="sm",
                                className="mt-auto",
                            )
                        ],
                    ),
                ],
                className="d-flex flex-column",
                style={
                    "height": CARD_HEIGHT
                },  # Adjusted height if needed due to button
            ),
            style={
                "width": CARD_WIDTH,
                "height": "auto",
            },  # Adjusted height to auto
            className="shadow-sm h-100",
        )

    @staticmethod
    def build_analysis_card(analysis: TFIDFAnalysis) -> dbc.Card:
        """Build the top activations analysis card."""
        # TF-IDF tokens section
        all_scores = [token["tf_idf"] for token in analysis.top_tokens]
        tfidf_badges = [
            dbc.Badge(
                [token["ability_name"], f" {token['tf_idf']:.2f}"],
                color=UIComponentBuilder.build_tfidf_badge_color(
                    token["tf_idf"], all_scores
                ),
                className="me-2 mb-2",
                pill=True,
                style={"fontSize": "0.9rem"},
            )
            for token in analysis.top_tokens
        ]

        # Weapons section
        weapon_bars = []
        colors = [
            "primary",
            "success",
            "warning",
            "danger",
        ]  # Different colors for each category
        for stat, name, column_name, color in zip(
            [
                analysis.top_weapons,
                analysis.sub_stats,
                analysis.special_stats,
                analysis.class_stats,
            ],
            ["Top Weapons", "Subs", "Specials", "Classes"],
            ["weapon_name", "sub", "special", "class"],
            colors,
        ):
            weapon_bars.append(
                dbc.Col(
                    html.Div(
                        [
                            html.H6(name, className="text-muted mb-2"),
                            *[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    item[column_name],
                                                    className="me-2",
                                                ),
                                                html.Small(
                                                    f"({item['percentage']:.0%})",
                                                    className="text-muted",
                                                ),
                                            ],
                                            className="d-flex justify-content-between",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Progress(
                                                    value=item["percentage"]
                                                    * 100,
                                                    color=color,
                                                    className="mb-1",
                                                    style={
                                                        "height": "15px",
                                                        "position": "relative",
                                                        "backgroundImage": "linear-gradient(to right, rgba(0,0,0,0.1) 1px, transparent 1px)",
                                                        "backgroundSize": "25% 100%",
                                                    },
                                                ),
                                                # Add tick marks
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            style={
                                                                "position": "absolute",
                                                                "left": f"{i * 25}%",
                                                                "top": "0",
                                                                "bottom": "0",
                                                                "width": "1px",
                                                                "backgroundColor": "rgba(0,0,0,0.2)",
                                                            }
                                                        )
                                                        for i in range(
                                                            1, 4
                                                        )  # Add 3 vertical lines at 25%, 50%, 75%
                                                    ],
                                                    style={
                                                        "position": "relative",
                                                        "height": "15px",
                                                    },
                                                ),
                                            ],
                                            style={"position": "relative"},
                                        ),
                                    ],
                                    className="mb-2",
                                )
                                for item in stat
                            ],
                        ],
                        className="mb-3",
                    ),
                    width=6,
                )
            )

        # Create a 2x2 grid layout
        stats_grid = dbc.Row(
            [
                dbc.Row(
                    [
                        weapon_bars[0],  # Top Weapons
                        weapon_bars[1],  # Subs
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        weapon_bars[2],  # Specials
                        weapon_bars[3],  # Classes
                    ],
                ),
            ]
        )

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        f"{analysis.feature_display_name} - Top Activations Analysis",
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6(
                                        "Top TF-IDF Tokens",
                                        className="text-muted mb-2",
                                    ),
                                    html.Div(
                                        tfidf_badges,
                                        className="d-flex flex-wrap",
                                    ),
                                ],
                                md=6,
                            ),
                            (
                                dbc.Col(
                                    [
                                        html.H6(
                                            "Activation Statistics",
                                            className="text-muted mb-2",
                                        ),
                                        stats_grid,
                                    ],
                                    md=6,
                                )
                                if analysis.top_weapons
                                else html.Div()
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        )

    @staticmethod
    def build_bin_section(
        bin_idx: int,
        min_act: float,
        max_act: float,
        samples: list[dict],
        count: int,
        weapon_names: dict[int, str],
        top_tfidf_tokens: set[str],
        inv_vocab: dict[str, str],
    ) -> html.Div:
        """Build a section for a histogram bin."""
        header_text = f"Bin {bin_idx[0]}: [{min_act:.3f}, {max_act:.3f}) - {count:,} examples"

        # Build example cards
        card_cols = []
        if count > 0:
            # Sample directly from the list
            import random

            samples_to_show = random.sample(
                samples, min(MAX_SAMPLES_PER_BIN, count)
            )

            for sample in samples_to_show:
                # Get weapon name from cache
                weapon_id = sample.get("weapon_id_token")
                weapon_name = weapon_names.get(weapon_id, f"Weapon {weapon_id}")

                card = UIComponentBuilder.build_example_card(
                    sample,
                    weapon_name,
                    sample["activation"],
                    top_tfidf_tokens,
                    inv_vocab,
                )
                card_cols.append(dbc.Col(card, width="auto", className="p-1"))

        return html.Div(
            [
                html.Div(header_text, className="fw-bold mb-2"),
                dbc.Row(
                    card_cols,
                    className="g-2 flex-wrap justify-content-start",
                ),
            ],
            className="p-3 mb-3 border rounded bg-light",
        )


class IntervalsGridRenderer:
    """Main renderer for the intervals grid component."""

    def __init__(self, dashboard_context: Any):
        self.context = dashboard_context
        self.db: FSDatabase = dashboard_context.db
        self.analyzer = TFIDFAnalyzer(
            self.context.inv_vocab,
            self.context.inv_weapon_vocab,
            generate_maps()[1],
            self.db.idf,
        )
        self._cache: Optional[FeatureAnalysisCache] = None

    def render(
        self, selected_feature_id: int, analysis_mode: str = "200"
    ) -> tuple[list[Any], str]:
        """Render the intervals grid for the selected feature.

        Args:
            selected_feature_id: The feature ID to render.
            analysis_mode: How many top activations to analyze.
                Options: "50", "200", "500", "1000", or "20pct" for percentage.
        """
        try:
            # Check if we need to reload data
            if (
                self._cache is None
                or self._cache.feature_id != selected_feature_id
            ):
                self._load_feature_data(selected_feature_id, analysis_mode)
            else:
                # Feature same but mode might have changed - re-run analysis
                self._perform_top_bins_analysis(analysis_mode)

            if self._cache is None:
                return [
                    html.P(f"No data found for feature {selected_feature_id}")
                ], ""

            sections = []

            # Add analysis card if available
            if self._cache.tfidf_analysis:
                sections.append(
                    UIComponentBuilder.build_analysis_card(
                        self._cache.tfidf_analysis
                    )
                )

            # Build bin sections
            sections.extend(self._build_all_bin_sections())

            return sections, ""

        except Exception as e:
            logger.error(f"Error rendering intervals grid: {e}", exc_info=True)
            return [], f"Error rendering intervals grid: {str(e)}"

    def _load_feature_data(
        self, feature_id: int, analysis_mode: str = "200"
    ) -> None:
        """Load and cache all data for a feature."""
        # Get data
        histogram_df = self.db.get_feature_histogram(feature_id)
        if histogram_df.is_empty():
            self._cache = None
            return

        # Limit activations for performance with large datasets
        # For efficient database, load more examples but still limit for performance
        max_activations = (
            5000  # Increased from 1000 to get better sampling
            if self.db.__class__.__name__ == "EfficientFSDatabase"
            else None
        )
        activations_df = self.db.get_feature_activations(
            feature_id, limit=max_activations
        )
        if activations_df.is_empty():
            self._cache = None
            return

        # Create cache
        self._cache = FeatureAnalysisCache(
            feature_id=feature_id,
            activations_df=activations_df,
            histogram_df=histogram_df,
        )

        # Perform TF-IDF analysis on top bins
        self._perform_top_bins_analysis(analysis_mode)

        # Cache weapon names
        self._cache_weapon_names()

    def _perform_top_bins_analysis(self, mode: str = "200") -> None:
        """Perform TF-IDF analysis on the TOP activations.

        Instead of using histogram bins (which may be pre-computed on different data),
        we directly take the top N activations sorted by activation value.
        This ensures we're always analyzing the actual highest activations.

        Args:
            mode: Analysis mode - either a number ("50", "200", "500", "1000")
                  or a percentage ("20pct" for top 20%).
        """
        total_samples = len(self._cache.activations_df)

        # Parse mode to determine how many samples to analyze
        if mode.endswith("pct"):
            # Percentage mode (e.g., "20pct" -> top 20%)
            pct = int(mode.replace("pct", ""))
            top_n = int(total_samples * pct / 100)
        else:
            # Fixed number mode
            top_n = min(int(mode), total_samples)

        # Ensure we have at least 1 sample
        top_n = max(1, top_n)

        top_activations = (
            self._cache.activations_df
            .sort("activation", descending=True)
            .head(top_n)
        )

        if top_activations.is_empty():
            logger.warning("No activations found for analysis!")
            return

        # Debug logging
        act_min = top_activations["activation"].min()
        act_max = top_activations["activation"].max()
        logger.info(
            f"Top activations analysis: using top {len(top_activations)} samples "
            f"(out of {total_samples} total, mode={mode}), "
            f"activation range: [{act_min:.4f}, {act_max:.4f}]"
        )

        # Analyze top activations
        self._cache.tfidf_analysis = self.analyzer.analyze(
            top_activations,
            getattr(self.context, "feature_labels_manager", None),
            self._cache.feature_id,
        )

    def _cache_weapon_names(self) -> None:
        """Cache weapon ID to name mappings for all weapons in the data."""
        # Get unique weapon IDs and create mapping in one operation
        weapon_mapping_df = (
            self._cache.activations_df.select("weapon_id_token")
            .unique()
            .with_columns(
                pl.col("weapon_id_token")
                .cast(pl.Utf8)
                .replace(self.analyzer.id_to_name_mapping)
                .alias("weapon_name")
            )
        )

        # Convert to dict for fast lookups
        self._cache.weapon_id_to_name = dict(
            zip(
                weapon_mapping_df["weapon_id_token"].to_list(),
                weapon_mapping_df["weapon_name"].to_list(),
            )
        )

    def _build_all_bin_sections(self) -> list[html.Div]:
        """Build sections for all histogram bins."""
        sections = []

        # Get top TF-IDF tokens for highlighting
        top_tfidf_tokens = (
            self._cache.tfidf_analysis.top_tfidf_token_names
            if self._cache.tfidf_analysis
            else set()
        )

        # Create a sorted list of bin edges for efficient binning
        histogram_sorted = self._cache.histogram_df.sort("bin_idx")

        # Create bin edges for cut operation
        cut_points = histogram_sorted["lower_bound"].to_list()[1:]
        bin_labels = histogram_sorted["bin_idx"].to_list()

        # Assign bins to all activations at once
        activations_with_bins = self._cache.activations_df.with_columns(
            pl.col("activation")
            .cut(cut_points, labels=[str(x) for x in bin_labels])
            .cast(pl.Int32)
            .alias("assigned_bin")
        )

        # Pre-filter bins with too few samples and get limited samples per bin
        # First get bin counts to filter empty bins
        bin_counts = activations_with_bins.group_by("assigned_bin").agg(
            pl.count().alias("count")
        )
        valid_bins = bin_counts.filter(pl.col("count") > 0)[
            "assigned_bin"
        ].to_list()

        # Then get limited samples for each valid bin
        grouped_samples = (
            activations_with_bins.filter(
                pl.col("assigned_bin").is_in(valid_bins)
            )
            .group_by("assigned_bin")
            .agg(
                [
                    pl.col("activation")
                    .head(MAX_SAMPLES_PER_BIN)
                    .alias("activation_samples"),
                    pl.col("weapon_id_token")
                    .head(MAX_SAMPLES_PER_BIN)
                    .alias("weapon_samples"),
                    pl.col("ability_input_tokens")
                    .head(MAX_SAMPLES_PER_BIN)
                    .alias("ability_samples"),
                    pl.col("activation").count().alias("count"),
                ]
            )
        )

        # Pre-compute histogram row lookups for valid bins
        bin_rows = {
            row[0]: (row[1], row[2])  # bin_idx -> (lower_bound, upper_bound)
            for row in histogram_sorted.filter(
                pl.col("bin_idx").is_in(valid_bins)
            ).iter_rows()
        }

        # Process each bin
        for row in grouped_samples.iter_rows():
            bin_idx = row[0]
            activation_samples = row[1]
            weapon_samples = row[2]
            ability_samples = row[3]
            count = row[4]
            # Combine samples into list of dicts
            bin_samples = [
                {
                    "activation": act,
                    "weapon_id_token": weap,
                    "ability_input_tokens": abil,
                }
                for act, weap, abil in zip(
                    activation_samples, weapon_samples, ability_samples
                )
            ]

            # Get pre-computed bin bounds
            lower_bound, upper_bound = bin_rows[bin_idx]

            # Build section for this bin
            section = UIComponentBuilder.build_bin_section(
                (bin_idx,),  # Wrap in tuple to match expected format
                lower_bound,
                upper_bound,
                bin_samples,
                count,
                self._cache.weapon_id_to_name,
                top_tfidf_tokens,
                self.context.inv_vocab,
            )
            sections.append((section, bin_idx))

        # Sort sections
        return [
            section
            for section, _ in sorted(sections, key=lambda x: x[1], reverse=True)
        ]


# ---------------------------------------------------------------------------
# Percentile Range Analysis
# ---------------------------------------------------------------------------


def get_feature_influence_data(feature_id: int) -> Optional[dict]:
    """Get influence data for a feature from the dashboard context."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        DASHBOARD_CONTEXT is None
        or not hasattr(DASHBOARD_CONTEXT, "influence_data")
        or DASHBOARD_CONTEXT.influence_data is None
    ):
        return None

    influence_df = DASHBOARD_CONTEXT.influence_data
    feature_data = influence_df[influence_df["feature_id"] == feature_id]

    if feature_data.empty:
        return None

    feature_row = feature_data.iloc[0]

    # Extract positive and negative influences
    positive = []
    negative = []

    for i in range(1, 31):  # Top 30
        pos_tok_col = f"+{i}_tok"
        pos_val_col = f"+{i}_val"
        neg_tok_col = f"-{i}_tok"
        neg_val_col = f"-{i}_val"

        if pos_tok_col in feature_row and feature_row[pos_tok_col]:
            positive.append(
                {"token": feature_row[pos_tok_col], "value": feature_row[pos_val_col]}
            )
        if neg_tok_col in feature_row and feature_row[neg_tok_col]:
            negative.append(
                {"token": feature_row[neg_tok_col], "value": feature_row[neg_val_col]}
            )

    return {"positive": positive, "negative": negative}


def compute_percentile_range_data(
    feature_id: int,
    feature_name: str,
    percentile_low: float,
    percentile_high: float,
    inv_vocab: dict[str, str],
    weapon_names: dict[int, str],
) -> Optional[PercentileRangeData]:
    """
    Compute percentile range data for a single feature.

    Uses raw Zarr data loading to get ALL non-zero activations for accurate
    percentile calculation based on ACTIVATION VALUE RANGE (not rank).

    Args:
        feature_id: The feature to analyze
        feature_name: Display name for the feature
        percentile_low: Lower percentile bound (e.g., 40.0)
        percentile_high: Upper percentile bound (e.g., 60.0)
        inv_vocab: Mapping from token ID to token name
        weapon_names: Mapping from weapon ID to weapon name

    Returns:
        PercentileRangeData with distributions and samples, or None if no data
    """
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if DASHBOARD_CONTEXT is None or DASHBOARD_CONTEXT.db is None:
        return None

    db = DASHBOARD_CONTEXT.db

    # Load ALL non-zero activations using raw data loading
    if hasattr(db, "_load_feature_activations"):
        _, all_activations = db._load_feature_activations(feature_id)
    else:
        all_activations = db.get_feature_activations(feature_id, limit=None)

    if all_activations is None or len(all_activations) == 0:
        return None

    # Get activation values to compute value-range percentiles
    act_col = "activation"
    activation_values = all_activations[act_col].to_numpy()

    # Compute percentile thresholds based on ACTIVATION VALUE RANGE
    # e.g., if activations range 0.0-1.0, top 10% = activations in 0.9-1.0
    min_act = float(activation_values.min())
    max_act = float(activation_values.max())
    act_range = max_act - min_act

    if act_range == 0:
        return None

    low_threshold = min_act + (percentile_low / 100) * act_range
    high_threshold = min_act + (percentile_high / 100) * act_range

    # Filter to examples within the activation value range
    range_df = all_activations.filter(
        (pl.col(act_col) >= low_threshold) & (pl.col(act_col) <= high_threshold)
    )

    if range_df.is_empty():
        return None

    # Sort by activation descending
    range_df = range_df.sort(act_col, descending=True)

    # Get column names
    weapon_col = "weapon_id" if "weapon_id" in range_df.columns else "weapon_id_token"
    ability_col = (
        "ability_tokens"
        if "ability_tokens" in range_df.columns
        else "ability_input_tokens"
    )

    # Compute activation stats
    activations = range_df[act_col].to_list()
    activation_min = min(activations)
    activation_max = max(activations)
    median_activation = float(np.median(activations))

    # Count token frequencies
    all_tokens = []
    for row in range_df.iter_rows(named=True):
        tokens = row.get(ability_col, [])
        if tokens:
            all_tokens.extend(tokens)

    token_counts = Counter(all_tokens)
    example_count = len(range_df)

    # Convert to distribution with percentages
    token_distribution = {}
    for token_id, count in token_counts.most_common(15):
        # Try direct lookup first, then string conversion as fallback
        token_name = inv_vocab.get(token_id) or inv_vocab.get(str(token_id), str(token_id))
        percentage = (count / example_count) * 100
        token_distribution[token_name] = (count, percentage)

    # Count weapon frequencies
    weapon_counts = Counter()
    for row in range_df.iter_rows(named=True):
        weapon_id = row.get(weapon_col)
        if weapon_id is not None:
            weapon_counts[weapon_id] += 1

    # Get weapon properties for hover info
    weapon_props_df = get_weapon_properties_df()
    weapon_props_dict = {}
    for row in weapon_props_df.iter_rows(named=True):
        weapon_props_dict[row["weapon_id"]] = {
            "sub": row.get("sub", ""),
            "special": row.get("special", ""),
            "class": row.get("class", ""),
        }

    # Get inv_weapon_vocab from context to convert token_id -> weapon_id_string
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    weapon_distribution = {}
    weapon_properties = {}
    for weapon_token_id, count in weapon_counts.most_common(10):
        weapon_name = (
            weapon_names.get(weapon_token_id)
            or weapon_names.get(str(weapon_token_id))
            or f"Weapon {weapon_token_id}"
        )
        percentage = (count / example_count) * 100
        weapon_distribution[weapon_name] = (count, percentage)

        # Get properties: convert token_id -> weapon_id_string -> properties
        weapon_id_str = inv_weapon_vocab.get(weapon_token_id, "")
        props = weapon_props_dict.get(weapon_id_str, {})
        weapon_properties[weapon_name] = props

    # Compute net influence at median activation
    influence_data = get_feature_influence_data(feature_id)
    net_influences = []
    if influence_data:
        for item in influence_data.get("positive", [])[:10]:
            net_influences.append(
                {
                    "token": item["token"],
                    "raw": item["value"],
                    "net": item["value"] * median_activation,
                    "direction": "positive",
                }
            )
        for item in influence_data.get("negative", [])[:10]:
            net_influences.append(
                {
                    "token": item["token"],
                    "raw": item["value"],
                    "net": item["value"] * median_activation,
                    "direction": "negative",
                }
            )

    # Sample examples (spread across the range)
    sample_indices = [
        0,
        len(range_df) // 4,
        len(range_df) // 2,
        3 * len(range_df) // 4,
        -1,
    ]
    sample_examples = []
    for idx in sample_indices:
        if idx < 0:
            idx = len(range_df) + idx
        if 0 <= idx < len(range_df):
            row = range_df.row(idx, named=True)
            weapon_id = row.get(weapon_col)
            weapon_name = (
                weapon_names.get(weapon_id)
                or weapon_names.get(str(weapon_id))
                or f"Weapon {weapon_id}"
            )
            ability_tokens = row.get(ability_col, [])
            ability_names = [
                inv_vocab.get(t) or inv_vocab.get(str(t), str(t))
                for t in ability_tokens[:5]
            ]
            sample_examples.append(
                {
                    "weapon": weapon_name,
                    "abilities": ability_names,
                    "activation": row.get(act_col, 0.0),
                }
            )

    return PercentileRangeData(
        feature_id=feature_id,
        feature_name=feature_name,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        activation_min=activation_min,
        activation_max=activation_max,
        example_count=example_count,
        token_distribution=token_distribution,
        weapon_distribution=weapon_distribution,
        weapon_properties=weapon_properties,
        median_activation=median_activation,
        net_influences=net_influences,
        sample_examples=sample_examples,
    )


def build_percentile_range_card(
    range_data: PercentileRangeData,
    title: str,
    color: str = "primary",
) -> dbc.Card:
    """Build UI card for a single percentile range."""
    # Token list
    token_items = []
    for token_name, (count, pct) in list(range_data.token_distribution.items())[:8]:
        token_items.append(html.Li(f"{token_name} ({pct:.0f}%)", className="small"))

    # Weapon list with sub/special hover
    weapon_items = []
    for weapon_name, (count, pct) in list(range_data.weapon_distribution.items())[:5]:
        props = range_data.weapon_properties.get(weapon_name, {})
        sub = props.get("sub", "")
        special = props.get("special", "")
        tooltip = f"Sub: {sub} | Special: {special}" if sub or special else ""
        weapon_items.append(
            html.Li(
                f"{weapon_name} ({pct:.0f}%)",
                className="small",
                title=tooltip,
                style={"cursor": "help"} if tooltip else {},
            )
        )

    # Net influence list
    pos_influences = [
        i for i in range_data.net_influences if i["direction"] == "positive"
    ][:5]
    neg_influences = [
        i for i in range_data.net_influences if i["direction"] == "negative"
    ][:5]

    influence_items = []
    for inf in pos_influences:
        influence_items.append(
            html.Li(
                f"+{inf['token']}: {inf['net']:+.3f}",
                className="small text-success",
            )
        )
    for inf in neg_influences:
        influence_items.append(
            html.Li(
                f"{inf['token']}: {inf['net']:+.3f}",
                className="small text-danger",
            )
        )

    # Sample examples
    example_items = []
    for ex in range_data.sample_examples[:3]:
        abilities_str = ", ".join(ex["abilities"][:3]) if ex["abilities"] else "—"
        example_items.append(
            html.Li(
                f"{ex['weapon']} - {abilities_str} (act: {ex['activation']:.2f})",
                className="small",
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H6(title, className="mb-0 text-white"),
                className=f"bg-{color}",
            ),
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.Span(
                                f"Activation: {range_data.activation_min:.2f} – {range_data.activation_max:.2f}",
                                className="me-3",
                            ),
                            html.Span(f"Examples: {range_data.example_count:,}"),
                        ],
                        className="small text-muted mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Strong("Top Tokens", className="small"),
                                    html.Ul(token_items, className="mb-0 ps-3"),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Strong("Top Weapons", className="small"),
                                    html.Ul(weapon_items, className="mb-0 ps-3"),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Strong(
                                        f"Net Influence (med={range_data.median_activation:.2f})",
                                        className="small",
                                    ),
                                    html.Ul(influence_items, className="mb-0 ps-3"),
                                ],
                                width=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Hr(className="my-2"),
                    html.Strong("Sample Builds", className="small"),
                    html.Ul(example_items, className="mb-0 ps-3"),
                ]
            ),
        ],
        className="h-100",
    )


def build_percentile_analysis_section(
    feature_id: int,
    feature_name: str,
    inv_vocab: dict[str, str],
    weapon_names: dict[int, str],
) -> html.Div:
    """Build the full percentile analysis section for a single feature."""
    # Compute data for each range
    top_range = compute_percentile_range_data(
        feature_id, feature_name, 90.0, 100.0, inv_vocab, weapon_names
    )
    middle_range = compute_percentile_range_data(
        feature_id, feature_name, 40.0, 60.0, inv_vocab, weapon_names
    )
    low_range = compute_percentile_range_data(
        feature_id, feature_name, 0.0, 20.0, inv_vocab, weapon_names
    )

    if not any([top_range, middle_range, low_range]):
        return html.Div(
            dbc.Alert(
                "No percentile data available for this feature.",
                color="warning",
            )
        )

    cards = []

    if top_range:
        cards.append(
            dbc.Col(
                build_percentile_range_card(
                    top_range,
                    f"Top 10% (90-100%)",
                    color="danger",
                ),
                md=4,
            )
        )

    if middle_range:
        cards.append(
            dbc.Col(
                build_percentile_range_card(
                    middle_range,
                    f"Middle 20% (40-60%)",
                    color="warning",
                ),
                md=4,
            )
        )

    if low_range:
        cards.append(
            dbc.Col(
                build_percentile_range_card(
                    low_range,
                    f"Low 20% (0-20%)",
                    color="secondary",
                ),
                md=4,
            )
        )

    return html.Div(
        [
            dbc.Alert(
                [
                    html.H6("Slider-Scale Hypothesis Test", className="mb-2"),
                    html.P(
                        "Comparing token/weapon distributions across activation VALUE ranges. "
                        "If this feature is slider-scale, higher activations should show "
                        "higher-investment tokens, while middle activations show moderate tokens.",
                        className="mb-0 small",
                    ),
                ],
                color="info",
                className="mb-3",
            ),
            dbc.Row(cards, className="g-3"),
        ]
    )


# ---------------------------------------------------------------------------
# Layout Component
# ---------------------------------------------------------------------------

intervals_grid_component = html.Div(
    [
        html.H4("Subsampled Intervals Grid", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Analysis Sample Size:", className="me-2"),
                        dbc.Select(
                            id="top-activations-mode",
                            options=[
                                {"label": "Top 50 examples", "value": "50"},
                                {"label": "Top 200 examples", "value": "200"},
                                {"label": "Top 500 examples", "value": "500"},
                                {"label": "Top 1000 examples", "value": "1000"},
                                {"label": "Top 20%", "value": "20pct"},
                            ],
                            value="200",
                            style={"width": "200px"},
                        ),
                    ],
                    width="auto",
                    className="d-flex align-items-center mb-3",
                ),
            ]
        ),
        # Percentile Range Analysis Section (collapsible)
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    dcc.Loading(
                        id="loading-percentile-analysis",
                        type="default",
                        children=html.Div(id="percentile-analysis-display"),
                    ),
                    title="Percentile Range Analysis (Slider-Scale Hypothesis)",
                    item_id="percentile-accordion",
                ),
            ],
            id="percentile-accordion-wrapper",
            start_collapsed=True,
            className="mb-4",
        ),
        # Intervals Grid Section
        dcc.Loading(
            id="loading-intervals-grid",
            type="default",
            children=html.Div(id="intervals-grid-display"),
        ),
        html.P(id="intervals-grid-error-message", style={"color": "red"}),
    ],
    id="intervals-grid-content",
    className="mb-4",
)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@callback(
    [
        Output("intervals-grid-display", "children"),
        Output("intervals-grid-error-message", "children"),
    ],
    [
        Input("feature-dropdown", "value"),
        Input("active-tab-store", "data"),
        Input("top-activations-mode", "value"),
    ],
)
def render_intervals_grid(
    selected_feature_id: int | None,
    active_tab: str | None,
    analysis_mode: str | None,
):
    """Callback to render the intervals grid based on selected feature."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Lazy loading: skip if tab is not active
    if active_tab != "tab-grid":
        return dash.no_update, dash.no_update

    if selected_feature_id is None:
        return [], "Select a feature."

    if (
        not DASHBOARD_CONTEXT
        or not hasattr(DASHBOARD_CONTEXT, "db")
        or not DASHBOARD_CONTEXT.db
    ):
        logger.debug("Dashboard context or database not available")
        return (
            [],
            "Error: Database context not available. Ensure data is loaded correctly.",
        )

    try:
        # Cache the renderer on DASHBOARD_CONTEXT to avoid recreating it
        # This preserves the internal cache between feature selections
        if not hasattr(DASHBOARD_CONTEXT, "_intervals_grid_renderer"):
            logger.info("Creating new IntervalsGridRenderer (first time)")
            DASHBOARD_CONTEXT._intervals_grid_renderer = IntervalsGridRenderer(
                DASHBOARD_CONTEXT
            )
        renderer = DASHBOARD_CONTEXT._intervals_grid_renderer
        return renderer.render(selected_feature_id, analysis_mode=analysis_mode or "200")
    except Exception as e:
        logger.error(f"Failed to render intervals grid: {e}", exc_info=True)
        return [], f"Error: {str(e)}"


@callback(
    Output("percentile-analysis-display", "children"),
    [
        Input("feature-dropdown", "value"),
        Input("active-tab-store", "data"),
        Input("percentile-accordion-wrapper", "active_item"),
    ],
)
def render_percentile_analysis(
    selected_feature_id: int | None,
    active_tab: str | None,
    accordion_active: str | None,
):
    """Callback to render the percentile range analysis section."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Lazy loading: skip if tab is not active or accordion is collapsed
    if active_tab != "tab-grid":
        return dash.no_update

    if accordion_active != "percentile-accordion":
        return html.Div(
            dbc.Alert(
                "Expand this section to load percentile analysis.",
                color="secondary",
                className="mb-0",
            )
        )

    if selected_feature_id is None:
        return html.Div(
            dbc.Alert("Select a feature to view percentile analysis.", color="info")
        )

    if (
        not DASHBOARD_CONTEXT
        or not hasattr(DASHBOARD_CONTEXT, "db")
        or not DASHBOARD_CONTEXT.db
    ):
        return html.Div(
            dbc.Alert("Database context not available.", color="warning")
        )

    try:
        # Get feature name
        feature_name = f"Feature {selected_feature_id}"
        if hasattr(DASHBOARD_CONTEXT, "feature_labels_manager"):
            labels_manager = DASHBOARD_CONTEXT.feature_labels_manager
            if labels_manager:
                feature_name = labels_manager.get_display_name(selected_feature_id)

        # Get weapon names mapping
        weapon_names = {}
        if hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab"):
            weapon_names = generate_weapon_name_mapping(
                DASHBOARD_CONTEXT.inv_weapon_vocab
            )

        return build_percentile_analysis_section(
            selected_feature_id,
            feature_name,
            DASHBOARD_CONTEXT.inv_vocab,
            weapon_names,
        )
    except Exception as e:
        logger.error(f"Failed to render percentile analysis: {e}", exc_info=True)
        return html.Div(
            dbc.Alert(f"Error: {str(e)}", color="danger")
        )


@callback(
    Output("ablation-primary-store", "data"),
    Output("analysis-tabs", "active_tab"),
    Input({"type": "select-ablation-primary", "index": dash.ALL}, "n_clicks"),
    State(
        {"type": "ablation-card-data-wrapper", "index": dash.ALL},
        "data-example",
    ),
    prevent_initial_call=True,
)
def update_ablation_primary_store(n_clicks_list, example_data_list):
    """Update the ablation-primary-store with data from the clicked example card."""
    ctx = dash.callback_context

    if not ctx.triggered_id:
        return dash.no_update, dash.no_update

    # Ensure triggered_id is a dict and has 'index' (expected for pattern-matching callbacks)
    if (
        not isinstance(ctx.triggered_id, dict)
        or "index" not in ctx.triggered_id
    ):
        return dash.no_update, dash.no_update

    clicked_button_index_str = ctx.triggered_id["index"]

    clicked_idx = -1
    # Find which button was clicked based on the index
    if ctx.inputs_list and isinstance(ctx.inputs_list[0], list):
        for i, input_def in enumerate(ctx.inputs_list[0]):
            if (
                isinstance(input_def.get("id"), dict)
                and input_def["id"].get("index") == clicked_button_index_str
            ):
                # Check if this specific button's n_clicks is what triggered the callback
                if n_clicks_list[i] is not None and n_clicks_list[i] > 0:
                    clicked_idx = i
                    break

    if clicked_idx != -1 and clicked_idx < len(example_data_list):
        raw_data_str = example_data_list[clicked_idx]
        if raw_data_str:
            try:
                example_data = json.loads(raw_data_str)
                data_to_store = {
                    "ability_input_tokens": example_data.get(
                        "ability_input_tokens"
                    ),
                    "weapon_id_token": example_data.get("weapon_id_token"),
                    "activation": example_data.get("activation"),
                }
                return (
                    data_to_store,
                    "tab-ablation",
                )  # Also switch to ablation tab
            except json.JSONDecodeError:
                return dash.no_update, dash.no_update

    return dash.no_update, dash.no_update


@callback(
    Output("full-sweep-build-queue", "data", allow_duplicate=True),
    Input({"type": "add-to-sweep-queue", "index": dash.ALL}, "n_clicks"),
    State(
        {"type": "ablation-card-data-wrapper", "index": dash.ALL},
        "data-example",
    ),
    State("full-sweep-build-queue", "data"),
    prevent_initial_call=True,
)
def add_example_to_sweep_queue(n_clicks_list, example_data_list, current_queue):
    """Add clicked example to the full sweep build queue."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    from splatnlp.dashboard.utils.converters import generate_weapon_name_mapping

    ctx = dash.callback_context

    if not ctx.triggered_id:
        return dash.no_update

    # Ensure triggered_id is a dict and has 'index'
    if (
        not isinstance(ctx.triggered_id, dict)
        or "index" not in ctx.triggered_id
    ):
        return dash.no_update

    clicked_button_index_str = ctx.triggered_id["index"]

    # Initialize queue if None
    if current_queue is None:
        current_queue = []

    # Find the matching data wrapper by index
    # The states_list contains the wrapper data in the same order as they appear in DOM
    raw_data_str = None
    if ctx.states_list and len(ctx.states_list) > 0:
        states = ctx.states_list[0] if isinstance(ctx.states_list[0], list) else ctx.states_list
        for i, state_def in enumerate(states):
            if isinstance(state_def, dict) and isinstance(state_def.get("id"), dict):
                if state_def["id"].get("index") == clicked_button_index_str:
                    raw_data_str = state_def.get("value")
                    break

    if raw_data_str:
        try:
            example_data = json.loads(raw_data_str)

            inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
            inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
            weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)

            weapon_id = example_data.get("weapon_id_token")
            ability_token_ids = example_data.get("ability_input_tokens", [])

            # Convert token IDs to names
            ability_tokens = [
                str(inv_vocab.get(token_id, f"Unknown_{token_id}"))
                for token_id in ability_token_ids
            ]

            # Get weapon name
            weapon_raw = inv_weapon_vocab.get(weapon_id, f"Unknown_{weapon_id}")
            weapon_name = weapon_name_mapping.get(weapon_id, weapon_raw)

            # Create description
            abilities_str = ", ".join(ability_tokens[:3])
            if len(ability_tokens) > 3:
                abilities_str += f"... (+{len(ability_tokens) - 3})"
            description = f"{abilities_str} + {weapon_name}"

            # Add to queue
            new_build = {
                "weapon_id": weapon_id,
                "weapon_name": weapon_name,
                "ability_tokens": ability_tokens,
                "description": description,
            }
            current_queue.append(new_build)
            return current_queue

        except json.JSONDecodeError:
            return dash.no_update

    return dash.no_update
