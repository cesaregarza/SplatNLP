"""
Feature Intervals Grid Dashboard Component

A refactored version of the intervals grid visualization for feature analysis.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
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
MAX_SAMPLES_PER_BIN = 14
TOP_WEAPONS_COUNT = 5
TOP_BINS_FOR_ANALYSIS = 8
CARD_WIDTH = "250px"
CARD_HEIGHT = "180px"


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
        """Calculate weapon frequencies in the dataframe."""
        total = df.select(pl.col("activation").sum()).item()

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

        # Group by weapon, count, and compute percentages in one go
        weapon_stats = (
            df.group_by("weapon_id_token")
            .agg(pl.col("activation").sum().alias("count"))
            .with_columns(
                [
                    pl.col("weapon_id_token")
                    .cast(pl.Utf8)
                    .replace(self.id_to_name_mapping)
                    .alias("weapon_name"),
                    (pl.col("count") / total).alias("percentage"),
                ]
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["weapon_name", "percentage"])
            .to_dicts()
        )

        sub_stats = (
            df.group_by("sub")
            .agg(pl.col("activation").sum().alias("count"))
            .with_columns(
                (pl.col("count") / total).alias("percentage"),
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["sub", "percentage"])
            .to_dicts()
        )

        special_stats = (
            df.group_by("special")
            .agg(pl.col("activation").sum().alias("count"))
            .with_columns(
                (pl.col("count") / total).alias("percentage"),
            )
            .sort("count", descending=True)
            .head(TOP_WEAPONS_COUNT)
            .select(["special", "percentage"])
            .to_dicts()
        )

        class_stats = (
            df.group_by("class")
            .agg(pl.col("activation").sum().alias("count"))
            .with_columns(
                (pl.col("count") / total).alias("percentage"),
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
                            "fontSize": "0.9rem",
                            "maxHeight": "calc(100% - 3rem)",
                        },  # Reserve space for title and activation
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
                            dbc.Button(
                                "Use as Primary for Ablation",
                                id={
                                    "type": "select-ablation-primary",
                                    "index": unique_example_id,
                                },
                                color="primary",
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

    def render(self, selected_feature_id: int) -> tuple[list[Any], str]:
        """Render the intervals grid for the selected feature."""
        try:
            # Check if we need to reload data
            if (
                self._cache is None
                or self._cache.feature_id != selected_feature_id
            ):
                self._load_feature_data(selected_feature_id)

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

    def _load_feature_data(self, feature_id: int) -> None:
        """Load and cache all data for a feature."""
        # Get data
        histogram_df = self.db.get_feature_histogram(feature_id)
        if histogram_df.is_empty():
            self._cache = None
            return

        activations_df = self.db.get_feature_activations(feature_id)
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
        self._perform_top_bins_analysis()

        # Cache weapon names
        self._cache_weapon_names()

    def _perform_top_bins_analysis(self) -> None:
        """Perform TF-IDF analysis on samples from top bins."""
        # Get bounds of top bins
        top_bins = (
            self._cache.histogram_df.sort("bin_idx", descending=True)
            .head(TOP_BINS_FOR_ANALYSIS)
            .select(["lower_bound", "upper_bound"])
        )

        # Convert bounds to lists for vectorized operation
        lower_bounds = top_bins["lower_bound"].to_list()
        upper_bounds = top_bins["upper_bound"].to_list()

        # Create filter for all top bins using any/or logic
        filter_expr = pl.any_horizontal(
            [
                pl.col("activation").is_between(lower, upper, closed="left")
                for lower, upper in zip(lower_bounds, upper_bounds)
            ]
        )

        # Get all samples from top bins in one operation
        combined_samples = self._cache.activations_df.filter(filter_expr)

        if combined_samples.is_empty():
            return

        # Analyze combined samples
        self._cache.tfidf_analysis = self.analyzer.analyze(
            combined_samples,
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
# Layout Component
# ---------------------------------------------------------------------------

intervals_grid_component = html.Div(
    [
        html.H4("Subsampled Intervals Grid", className="mb-3"),
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
    Input("feature-dropdown", "value"),
)
def render_intervals_grid(selected_feature_id: int | None):
    """Callback to render the intervals grid based on selected feature."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

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
        renderer = IntervalsGridRenderer(DASHBOARD_CONTEXT)
        return renderer.render(selected_feature_id)
    except Exception as e:
        logger.error(f"Failed to render intervals grid: {e}", exc_info=True)
        return [], f"Error: {str(e)}"


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
