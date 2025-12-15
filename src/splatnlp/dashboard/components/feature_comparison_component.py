"""
Feature Comparison Dashboard Component

Allows comparing multiple SAE features side-by-side with:
- TF-IDF analysis and weapon stats for each feature
- Overlap analysis (Venn diagram for 2 features)
- Shared examples metadata
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dcc,
    html,
    no_update,
)

logger = logging.getLogger(__name__)

# Import shared utilities from intervals_grid_component
from splatnlp.dashboard.components.intervals_grid_component import (
    TFIDFAnalysis,
    TFIDFAnalyzer,
)

# Configuration
MAX_COMPARISON_FEATURES = 10  # Allow up to 10 features for comparison
DEFAULT_ACTIVATION_LIMIT = 100  # Top 100 examples per feature for matrix
MAX_CARDS_PER_ROW = 4  # Maximum cards in a single row before wrapping


@dataclass
class ComparisonFeatureData:
    """Holds all comparison data for a single feature."""

    feature_id: int
    display_name: str
    tfidf_analysis: Optional[TFIDFAnalysis]
    activations_df: pl.DataFrame
    example_indices: set[int] = field(default_factory=set)


@dataclass
class OverlapAnalysis:
    """Results of overlap analysis between features."""

    shared_indices: set[int]
    unique_indices: dict[int, set[int]]  # feature_id -> unique indices
    shared_count: int
    feature_counts: dict[int, int]  # feature_id -> total count


@dataclass
class InfluenceBreakdown:
    """Breakdown of activation-weighted influence for a token."""

    token: str
    raw_influence_a: float  # Per-unit influence for feature A
    activation_a: float  # Activation value for feature A on this example
    net_influence_a: float  # raw × activation
    raw_influence_b: float  # Per-unit influence for feature B
    activation_b: float  # Activation value for feature B on this example
    net_influence_b: float  # raw × activation
    difference: float  # net_a - net_b
    agreement: str  # "agree", "conflict", or "neutral"


@dataclass
class AggregateInfluence:
    """Aggregated net influence for a token across all shared examples."""

    token: str
    total_net_a: float  # Sum of (activation_a × raw_influence_a) across examples
    total_net_b: float  # Sum of (activation_b × raw_influence_b) across examples
    raw_influence_a: float  # Raw per-unit influence for feature A
    raw_influence_b: float  # Raw per-unit influence for feature B
    example_count: int  # Number of shared examples contributing
    agreement: str  # "agree", "conflict", or "neutral"


@dataclass
class FeatureRelationship:
    """Complete relationship analysis between two features."""

    feature_a_id: int
    feature_b_id: int
    feature_a_name: str
    feature_b_name: str
    shared_example_count: int
    # Activation correlation
    activation_correlation: float
    activation_pairs: list[tuple[float, float]]  # (act_a, act_b) for scatter
    # Aggregate influences
    aggregate_influences: list[AggregateInfluence]
    # Summary stats
    tokens_both_influence: int
    tokens_only_a: int
    tokens_only_b: int
    agreement_count: int
    conflict_count: int
    neutral_count: int


@dataclass
class PercentileRangeData:
    """Data for a single percentile range of a feature."""

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
    # Net influence using median activation
    median_activation: float
    net_influences: list[dict]  # [{token, raw, net}, ...]
    # Sample examples
    sample_examples: list[dict]  # [{weapon, abilities, activation}, ...]


@dataclass
class PercentileComparison:
    """Comparison of two features across percentile ranges."""

    feature_a_id: int
    feature_b_id: int
    feature_a_name: str
    feature_b_name: str
    top_range_a: PercentileRangeData  # 90-100%
    top_range_b: PercentileRangeData
    middle_range_a: PercentileRangeData  # 40-60%
    middle_range_b: PercentileRangeData


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

feature_comparison_component = html.Div(
    [
        html.H4("Feature Comparison", className="mb-3"),
        html.P(
            "Compare the selected feature with other features to analyze "
            "co-activation patterns and shared characteristics.",
            className="text-muted mb-3",
        ),
        # Stores
        dcc.Store(id="comparison-features-store", storage_type="session"),
        dcc.Store(id="comparison-data-cache", storage_type="memory"),
        # Feature Selection Area
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Primary Feature:",
                                        className="fw-bold",
                                    ),
                                    html.Div(
                                        id="primary-feature-display",
                                        className="text-primary fs-5",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Compare with:",
                                        className="fw-bold",
                                    ),
                                    dcc.Dropdown(
                                        id="comparison-features-dropdown",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="Select features to compare...",
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                width=8,
                            ),
                        ],
                        className="mb-2",
                    ),
                ]
            ),
            className="mb-3",
        ),
        # Comparison Results
        dcc.Loading(
            id="loading-comparison",
            type="default",
            children=[
                # Side-by-side comparison cards
                html.Div(id="comparison-cards-container", className="mb-4"),
                # Overlap matrix (n×n heatmap)
                html.Div(id="overlap-matrix-container", className="mb-4"),
                # Overlap analysis (summary table)
                html.Div(id="overlap-analysis-container", className="mb-4"),
                # Venn sub-selection (shown when 3+ features)
                html.Div(id="venn-subselect-container", className="mb-4"),
            ],
        ),
        html.Div(id="comparison-error-message", style={"color": "red"}),
        # Modal for influence analysis when clicking shared examples
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Activation-Weighted Influence Analysis"),
                    close_button=True,
                ),
                dbc.ModalBody(id="influence-analysis-content"),
            ],
            id="influence-analysis-modal",
            size="xl",
            is_open=False,
            scrollable=True,
        ),
        # Modal for feature relationship analysis (aggregate view)
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Feature Relationship Analysis"),
                    close_button=True,
                ),
                dbc.ModalBody(id="relationship-analysis-content"),
            ],
            id="relationship-analysis-modal",
            size="xl",
            is_open=False,
            scrollable=True,
        ),
        # Modal for percentile range analysis
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Percentile Range Analysis"),
                    close_button=True,
                ),
                dbc.ModalBody(id="percentile-analysis-content"),
            ],
            id="percentile-analysis-modal",
            size="xl",
            is_open=False,
            scrollable=True,
        ),
    ],
    id="feature-comparison-content",
    className="mb-4",
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def compute_influence_breakdown(
    feature_a_id: int,
    feature_b_id: int,
    activation_a: float,
    activation_b: float,
) -> list[InfluenceBreakdown]:
    """
    Compute activation-weighted influence breakdown for two features.

    Returns list of InfluenceBreakdown for all tokens influenced by either feature,
    sorted by absolute difference in net influence.
    """
    # Get influence data for both features
    influence_a = get_feature_influence_data(feature_a_id)
    influence_b = get_feature_influence_data(feature_b_id)

    if not influence_a and not influence_b:
        return []

    # Build token -> influence value mappings
    def build_influence_map(influence_data: Optional[dict]) -> dict[str, float]:
        if not influence_data:
            return {}
        result = {}
        for item in influence_data.get("positive", []):
            result[item["token"]] = item["value"]
        for item in influence_data.get("negative", []):
            result[item["token"]] = item["value"]
        return result

    influence_map_a = build_influence_map(influence_a)
    influence_map_b = build_influence_map(influence_b)

    # Get union of all tokens
    all_tokens = set(influence_map_a.keys()) | set(influence_map_b.keys())

    # Build breakdown for each token
    breakdowns = []
    for token in all_tokens:
        raw_a = influence_map_a.get(token, 0.0)
        raw_b = influence_map_b.get(token, 0.0)
        net_a = raw_a * activation_a
        net_b = raw_b * activation_b
        diff = net_a - net_b

        # Determine agreement status
        if abs(net_a) < 0.001 or abs(net_b) < 0.001:
            agreement = "neutral"
        elif (net_a > 0 and net_b > 0) or (net_a < 0 and net_b < 0):
            agreement = "agree"
        else:
            agreement = "conflict"

        breakdowns.append(
            InfluenceBreakdown(
                token=token,
                raw_influence_a=raw_a,
                activation_a=activation_a,
                net_influence_a=net_a,
                raw_influence_b=raw_b,
                activation_b=activation_b,
                net_influence_b=net_b,
                difference=diff,
                agreement=agreement,
            )
        )

    # Sort by absolute difference (most different first)
    breakdowns.sort(key=lambda x: abs(x.difference), reverse=True)

    return breakdowns


def build_influence_breakdown_table(
    breakdowns: list[InfluenceBreakdown],
    feature_a_name: str,
    feature_b_name: str,
) -> html.Div:
    """Build table showing raw vs net influence comparison."""
    if not breakdowns:
        return html.Div("No influence data available.", className="text-muted")

    # Truncate names for headers
    a_short = (
        (feature_a_name[:10] + "…")
        if len(feature_a_name) > 10
        else feature_a_name
    )
    b_short = (
        (feature_b_name[:10] + "…")
        if len(feature_b_name) > 10
        else feature_b_name
    )

    # Build rows - show top 30
    rows = []
    for bd in breakdowns[:30]:
        # Color code based on agreement
        if bd.agreement == "agree":
            status_badge = html.Span("✓ Agree", className="badge bg-success")
        elif bd.agreement == "conflict":
            status_badge = html.Span(
                "⚡ Conflict", className="badge bg-warning text-dark"
            )
        else:
            status_badge = html.Span(
                "— Neutral", className="badge bg-secondary"
            )

        # Color code difference
        diff_class = (
            "text-success"
            if bd.difference > 0
            else "text-danger" if bd.difference < 0 else ""
        )

        rows.append(
            html.Tr(
                [
                    html.Td(bd.token, className="small fw-bold"),
                    html.Td(
                        f"{bd.raw_influence_a:.4f}", className="small text-end"
                    ),
                    html.Td(
                        f"{bd.net_influence_a:.4f}",
                        className="small text-end fw-bold",
                    ),
                    html.Td(
                        f"{bd.raw_influence_b:.4f}", className="small text-end"
                    ),
                    html.Td(
                        f"{bd.net_influence_b:.4f}",
                        className="small text-end fw-bold",
                    ),
                    html.Td(
                        f"{bd.difference:+.4f}",
                        className=f"small text-end {diff_class}",
                    ),
                    html.Td(status_badge, className="small text-center"),
                ]
            )
        )

    return html.Div(
        [
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Token", className="small"),
                                    html.Th(
                                        f"Raw {a_short}",
                                        className="small text-end",
                                        title=f"Raw influence for {feature_a_name}",
                                    ),
                                    html.Th(
                                        f"Net {a_short}",
                                        className="small text-end",
                                        title=f"Net influence for {feature_a_name}",
                                    ),
                                    html.Th(
                                        f"Raw {b_short}",
                                        className="small text-end",
                                        title=f"Raw influence for {feature_b_name}",
                                    ),
                                    html.Th(
                                        f"Net {b_short}",
                                        className="small text-end",
                                        title=f"Net influence for {feature_b_name}",
                                    ),
                                    html.Th(
                                        "Diff",
                                        className="small text-end",
                                        title="Net A - Net B",
                                    ),
                                    html.Th(
                                        "Status", className="small text-center"
                                    ),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="table table-sm table-striped",
                ),
                style={"maxHeight": "400px", "overflowY": "auto"},
            ),
            html.P(
                f"Showing {min(30, len(breakdowns))} of {len(breakdowns)} influenced tokens",
                className="text-muted small mt-2",
            ),
        ]
    )


def build_influence_comparison_chart(
    breakdowns: list[InfluenceBreakdown],
    feature_a_name: str,
    feature_b_name: str,
) -> go.Figure:
    """Build grouped bar chart comparing net contributions."""
    if not breakdowns:
        fig = go.Figure()
        fig.update_layout(title="No influence data available")
        return fig

    # Take top 15 by absolute net influence (either feature)
    top_breakdowns = sorted(
        breakdowns,
        key=lambda x: max(abs(x.net_influence_a), abs(x.net_influence_b)),
        reverse=True,
    )[:15]

    # Reverse for horizontal bar chart (top at top)
    top_breakdowns = list(reversed(top_breakdowns))

    tokens = [bd.token for bd in top_breakdowns]
    net_a = [bd.net_influence_a for bd in top_breakdowns]
    net_b = [bd.net_influence_b for bd in top_breakdowns]

    # Truncate names for legend
    a_short = (
        (feature_a_name[:15] + "…")
        if len(feature_a_name) > 15
        else feature_a_name
    )
    b_short = (
        (feature_b_name[:15] + "…")
        if len(feature_b_name) > 15
        else feature_b_name
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=tokens,
            x=net_a,
            name=a_short,
            orientation="h",
            marker_color="rgba(66, 133, 244, 0.8)",
            text=[f"{v:.3f}" for v in net_a],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            y=tokens,
            x=net_b,
            name=b_short,
            orientation="h",
            marker_color="rgba(234, 67, 53, 0.8)",
            text=[f"{v:.3f}" for v in net_b],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Net Influence Comparison (activation × raw influence)",
        barmode="group",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        xaxis_title="Net Influence",
        yaxis_title="Token",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig


def compute_feature_relationship(
    feature_a_id: int,
    feature_b_id: int,
    feature_a_name: str,
    feature_b_name: str,
    shared_indices: set[int],
    examples_a: dict[int, dict],
    examples_b: dict[int, dict],
) -> FeatureRelationship:
    """
    Compute aggregate relationship analysis between two features.

    Aggregates net influences (activation × raw influence) across all shared examples.
    """
    import numpy as np

    # Get influence data for both features
    influence_a = get_feature_influence_data(feature_a_id)
    influence_b = get_feature_influence_data(feature_b_id)

    # Build token -> raw influence mappings
    def build_influence_map(influence_data: Optional[dict]) -> dict[str, float]:
        if not influence_data:
            return {}
        result = {}
        for item in influence_data.get("positive", []):
            result[item["token"]] = item["value"]
        for item in influence_data.get("negative", []):
            result[item["token"]] = item["value"]
        return result

    influence_map_a = build_influence_map(influence_a)
    influence_map_b = build_influence_map(influence_b)

    # Collect activation pairs for correlation
    activation_pairs = []
    for idx in shared_indices:
        ex_a = examples_a.get(idx)
        ex_b = examples_b.get(idx)
        if ex_a and ex_b:
            act_a = ex_a.get("activation", 0.0)
            act_b = ex_b.get("activation", 0.0)
            activation_pairs.append((act_a, act_b))

    # Compute correlation
    if len(activation_pairs) >= 2:
        acts_a = [p[0] for p in activation_pairs]
        acts_b = [p[1] for p in activation_pairs]
        if np.std(acts_a) > 0 and np.std(acts_b) > 0:
            activation_correlation = float(np.corrcoef(acts_a, acts_b)[0, 1])
        else:
            activation_correlation = 0.0
    else:
        activation_correlation = 0.0

    # Get all tokens influenced by either feature
    tokens_a = set(influence_map_a.keys())
    tokens_b = set(influence_map_b.keys())
    tokens_both = tokens_a & tokens_b
    tokens_only_a = tokens_a - tokens_b
    tokens_only_b = tokens_b - tokens_a
    all_tokens = tokens_a | tokens_b

    # Compute aggregate net influences
    aggregate_influences = []
    agreement_count = 0
    conflict_count = 0
    neutral_count = 0

    for token in all_tokens:
        raw_a = influence_map_a.get(token, 0.0)
        raw_b = influence_map_b.get(token, 0.0)

        # Sum net influence across all shared examples
        total_net_a = 0.0
        total_net_b = 0.0
        count = 0

        for idx in shared_indices:
            ex_a = examples_a.get(idx)
            ex_b = examples_b.get(idx)
            if ex_a and ex_b:
                act_a = ex_a.get("activation", 0.0)
                act_b = ex_b.get("activation", 0.0)
                total_net_a += raw_a * act_a
                total_net_b += raw_b * act_b
                count += 1

        # Determine agreement
        threshold = 0.01 * count if count > 0 else 0.01
        if abs(total_net_a) < threshold or abs(total_net_b) < threshold:
            agreement = "neutral"
            neutral_count += 1
        elif (total_net_a > 0 and total_net_b > 0) or (
            total_net_a < 0 and total_net_b < 0
        ):
            agreement = "agree"
            agreement_count += 1
        else:
            agreement = "conflict"
            conflict_count += 1

        aggregate_influences.append(
            AggregateInfluence(
                token=token,
                total_net_a=total_net_a,
                total_net_b=total_net_b,
                raw_influence_a=raw_a,
                raw_influence_b=raw_b,
                example_count=count,
                agreement=agreement,
            )
        )

    # Sort by combined absolute influence
    aggregate_influences.sort(
        key=lambda x: abs(x.total_net_a) + abs(x.total_net_b),
        reverse=True,
    )

    return FeatureRelationship(
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
        feature_a_name=feature_a_name,
        feature_b_name=feature_b_name,
        shared_example_count=len(shared_indices),
        activation_correlation=activation_correlation,
        activation_pairs=activation_pairs,
        aggregate_influences=aggregate_influences,
        tokens_both_influence=len(tokens_both),
        tokens_only_a=len(tokens_only_a),
        tokens_only_b=len(tokens_only_b),
        agreement_count=agreement_count,
        conflict_count=conflict_count,
        neutral_count=neutral_count,
    )


def build_activation_scatter(relationship: FeatureRelationship) -> go.Figure:
    """Build scatter plot of activation pairs with correlation."""
    if not relationship.activation_pairs:
        fig = go.Figure()
        fig.update_layout(title="No shared examples")
        return fig

    acts_a = [p[0] for p in relationship.activation_pairs]
    acts_b = [p[1] for p in relationship.activation_pairs]

    # Truncate names
    a_short = (
        (relationship.feature_a_name[:20] + "…")
        if len(relationship.feature_a_name) > 20
        else relationship.feature_a_name
    )
    b_short = (
        (relationship.feature_b_name[:20] + "…")
        if len(relationship.feature_b_name) > 20
        else relationship.feature_b_name
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=acts_a,
            y=acts_b,
            mode="markers",
            marker=dict(
                size=6,
                color="rgba(66, 133, 244, 0.6)",
                line=dict(width=1, color="rgba(66, 133, 244, 0.9)"),
            ),
            hovertemplate=f"{a_short}: %{{x:.3f}}<br>{b_short}: %{{y:.3f}}<extra></extra>",
        )
    )

    # Add trend line if correlation is meaningful
    if abs(relationship.activation_correlation) > 0.1 and len(acts_a) > 2:
        import numpy as np

        z = np.polyfit(acts_a, acts_b, 1)
        p = np.poly1d(z)
        x_line = [min(acts_a), max(acts_a)]
        y_line = [p(x) for x in x_line]
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="rgba(234, 67, 53, 0.7)", dash="dash"),
                name=f"r = {relationship.activation_correlation:.3f}",
            )
        )

    fig.update_layout(
        title=f"Activation Correlation: r = {relationship.activation_correlation:.3f}",
        xaxis_title=a_short,
        yaxis_title=b_short,
        height=300,
        margin=dict(l=60, r=20, t=40, b=50),
        showlegend=False,
    )

    return fig


def build_aggregate_influence_chart(
    relationship: FeatureRelationship,
) -> go.Figure:
    """Build grouped bar chart of aggregate net influences."""
    if not relationship.aggregate_influences:
        fig = go.Figure()
        fig.update_layout(title="No influence data")
        return fig

    # Take top 20 by combined influence
    top_influences = relationship.aggregate_influences[:20]

    # Reverse for display (top at top)
    top_influences = list(reversed(top_influences))

    tokens = [ai.token for ai in top_influences]
    total_a = [ai.total_net_a for ai in top_influences]
    total_b = [ai.total_net_b for ai in top_influences]

    # Color based on agreement
    colors_a = []
    colors_b = []
    for ai in top_influences:
        if ai.agreement == "conflict":
            colors_a.append("rgba(234, 67, 53, 0.8)")  # Red for conflict
            colors_b.append("rgba(234, 67, 53, 0.8)")
        elif ai.agreement == "agree":
            colors_a.append("rgba(52, 168, 83, 0.8)")  # Green for agree
            colors_b.append("rgba(52, 168, 83, 0.8)")
        else:
            colors_a.append("rgba(158, 158, 158, 0.8)")  # Gray for neutral
            colors_b.append("rgba(158, 158, 158, 0.8)")

    # Truncate names
    a_short = (
        (relationship.feature_a_name[:15] + "…")
        if len(relationship.feature_a_name) > 15
        else relationship.feature_a_name
    )
    b_short = (
        (relationship.feature_b_name[:15] + "…")
        if len(relationship.feature_b_name) > 15
        else relationship.feature_b_name
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=tokens,
            x=total_a,
            name=a_short,
            orientation="h",
            marker_color="rgba(66, 133, 244, 0.8)",
            text=[f"{v:.2f}" for v in total_a],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            y=tokens,
            x=total_b,
            name=b_short,
            orientation="h",
            marker_color="rgba(251, 188, 5, 0.8)",
            text=[f"{v:.2f}" for v in total_b],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Aggregate Net Influence (summed across {relationship.shared_example_count} shared examples)",
        barmode="group",
        height=500,
        margin=dict(l=150, r=80, t=50, b=50),
        xaxis_title="Total Net Influence (Σ activation × raw)",
        yaxis_title="Token",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig


def build_differentiators_section(
    relationship: FeatureRelationship,
) -> html.Div:
    """Build section showing key differentiators between features."""
    # Get conflicts (opposite signs)
    conflicts = [
        ai
        for ai in relationship.aggregate_influences
        if ai.agreement == "conflict"
    ]
    # Get agreements (same sign, non-neutral)
    agreements = [
        ai
        for ai in relationship.aggregate_influences
        if ai.agreement == "agree"
    ]
    # Get tokens unique to each feature
    unique_a = [
        ai
        for ai in relationship.aggregate_influences
        if ai.raw_influence_b == 0 and ai.raw_influence_a != 0
    ]
    unique_b = [
        ai
        for ai in relationship.aggregate_influences
        if ai.raw_influence_a == 0 and ai.raw_influence_b != 0
    ]

    # Truncate names
    a_short = (
        (relationship.feature_a_name[:15] + "…")
        if len(relationship.feature_a_name) > 15
        else relationship.feature_a_name
    )
    b_short = (
        (relationship.feature_b_name[:15] + "…")
        if len(relationship.feature_b_name) > 15
        else relationship.feature_b_name
    )

    def make_token_list(
        items: list[AggregateInfluence], show_both: bool = True
    ) -> html.Ul:
        if not items:
            return html.P("None", className="text-muted small")
        list_items = []
        for ai in items[:8]:  # Top 8
            if show_both:
                list_items.append(
                    html.Li(
                        f"{ai.token}: A={ai.total_net_a:+.2f}, B={ai.total_net_b:+.2f}",
                        className="small",
                    )
                )
            else:
                val = (
                    ai.total_net_a
                    if ai.raw_influence_a != 0
                    else ai.total_net_b
                )
                list_items.append(
                    html.Li(f"{ai.token}: {val:+.2f}", className="small")
                )
        return html.Ul(list_items, className="mb-0")

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6(
                                f"🔵 Unique to {a_short}",
                                className="text-primary",
                            ),
                            make_token_list(unique_a, show_both=False),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H6(
                                f"🟡 Unique to {b_short}",
                                className="text-warning",
                            ),
                            make_token_list(unique_b, show_both=False),
                        ],
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6(
                                "✓ Strongest Agreements",
                                className="text-success",
                            ),
                            make_token_list(agreements[:8]),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H6(
                                "⚡ Strongest Conflicts",
                                className="text-danger",
                            ),
                            make_token_list(conflicts[:8]),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


def format_relationship_markdown(relationship: FeatureRelationship) -> str:
    """Format relationship analysis as Markdown for Obsidian."""
    # Summary statistics
    total_tokens = (
        relationship.tokens_both_influence
        + relationship.tokens_only_a
        + relationship.tokens_only_b
    )
    conflict_rate = (
        relationship.conflict_count
        / max(1, relationship.agreement_count + relationship.conflict_count)
        * 100
    )

    # Interpretation
    if relationship.activation_correlation > 0.7:
        corr_interpretation = "highly correlated (often fire together)"
    elif relationship.activation_correlation > 0.3:
        corr_interpretation = "moderately correlated"
    elif relationship.activation_correlation > -0.3:
        corr_interpretation = "independent (fire separately)"
    elif relationship.activation_correlation > -0.7:
        corr_interpretation = "moderately anti-correlated"
    else:
        corr_interpretation = "highly anti-correlated (mutually exclusive)"

    if conflict_rate > 50:
        conflict_interpretation = "These features often compete - they push outputs in opposite directions."
    elif conflict_rate > 20:
        conflict_interpretation = (
            "These features have some competition but mostly cooperate."
        )
    else:
        conflict_interpretation = (
            "These features largely cooperate - they reinforce similar outputs."
        )

    # Get differentiators
    conflicts = [
        ai
        for ai in relationship.aggregate_influences
        if ai.agreement == "conflict"
    ]
    agreements = [
        ai
        for ai in relationship.aggregate_influences
        if ai.agreement == "agree"
    ]
    unique_a = [
        ai
        for ai in relationship.aggregate_influences
        if ai.raw_influence_b == 0 and ai.raw_influence_a != 0
    ]
    unique_b = [
        ai
        for ai in relationship.aggregate_influences
        if ai.raw_influence_a == 0 and ai.raw_influence_b != 0
    ]

    # Build markdown
    lines = [
        f"## Feature Relationship: {relationship.feature_a_name} vs {relationship.feature_b_name}",
        "",
        f"Analyzed across **{relationship.shared_example_count:,}** shared examples.",
        "",
        "### Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Activation Correlation | {relationship.activation_correlation:.2f} ({corr_interpretation}) |",
        f"| Conflict Rate | {conflict_rate:.0f}% ({relationship.conflict_count} conflicts / {relationship.agreement_count} agreements) |",
        f"| Tokens Influenced | {total_tokens} ({relationship.tokens_both_influence} shared, {relationship.tokens_only_a} unique to A, {relationship.tokens_only_b} unique to B) |",
        "",
        f"> {conflict_interpretation}",
        "",
        "### Token Influence Breakdown",
        "",
        f"- ✓ **Agreements**: {relationship.agreement_count}",
        f"- ⚡ **Conflicts**: {relationship.conflict_count}",
        f"- — **Neutral**: {relationship.neutral_count}",
        "",
    ]

    # Unique to A
    if unique_a:
        lines.append(f"#### Unique to {relationship.feature_a_name}")
        lines.append("")
        for ai in unique_a[:8]:
            lines.append(f"- `{ai.token}`: {ai.total_net_a:+.2f}")
        lines.append("")

    # Unique to B
    if unique_b:
        lines.append(f"#### Unique to {relationship.feature_b_name}")
        lines.append("")
        for ai in unique_b[:8]:
            lines.append(f"- `{ai.token}`: {ai.total_net_b:+.2f}")
        lines.append("")

    # Strongest agreements
    if agreements:
        lines.append("#### Strongest Agreements")
        lines.append("")
        for ai in agreements[:8]:
            lines.append(
                f"- `{ai.token}`: A={ai.total_net_a:+.2f}, B={ai.total_net_b:+.2f}"
            )
        lines.append("")

    # Strongest conflicts
    if conflicts:
        lines.append("#### Strongest Conflicts")
        lines.append("")
        for ai in conflicts[:8]:
            lines.append(
                f"- `{ai.token}`: A={ai.total_net_a:+.2f}, B={ai.total_net_b:+.2f}"
            )
        lines.append("")

    # Top influences table
    lines.append("### Top Aggregate Influences")
    lines.append("")
    lines.append("| Token | Net A | Net B | Status |")
    lines.append("|-------|-------|-------|--------|")
    for ai in relationship.aggregate_influences[:15]:
        status = (
            "✓ Agree"
            if ai.agreement == "agree"
            else "⚡ Conflict" if ai.agreement == "conflict" else "—"
        )
        lines.append(
            f"| `{ai.token}` | {ai.total_net_a:+.2f} | {ai.total_net_b:+.2f} | {status} |"
        )
    lines.append("")

    return "\n".join(lines)


def build_relationship_analysis_content(
    relationship: FeatureRelationship,
) -> html.Div:
    """Build the full modal content for feature relationship analysis."""
    # Summary statistics
    total_tokens = (
        relationship.tokens_both_influence
        + relationship.tokens_only_a
        + relationship.tokens_only_b
    )
    conflict_rate = (
        relationship.conflict_count
        / max(1, relationship.agreement_count + relationship.conflict_count)
        * 100
    )

    # Interpretation
    if relationship.activation_correlation > 0.7:
        corr_interpretation = "highly correlated (often fire together)"
    elif relationship.activation_correlation > 0.3:
        corr_interpretation = "moderately correlated"
    elif relationship.activation_correlation > -0.3:
        corr_interpretation = "independent (fire separately)"
    elif relationship.activation_correlation > -0.7:
        corr_interpretation = "moderately anti-correlated"
    else:
        corr_interpretation = "highly anti-correlated (mutually exclusive)"

    if conflict_rate > 50:
        conflict_interpretation = "These features often compete - they push outputs in opposite directions."
    elif conflict_rate > 20:
        conflict_interpretation = (
            "These features have some competition but mostly cooperate."
        )
    else:
        conflict_interpretation = (
            "These features largely cooperate - they reinforce similar outputs."
        )

    # Generate markdown for clipboard
    markdown_text = format_relationship_markdown(relationship)

    return html.Div(
        [
            # Header summary with copy button
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H5(
                                        [
                                            relationship.feature_a_name,
                                            html.Span(
                                                " vs ",
                                                className="text-muted mx-2",
                                            ),
                                            relationship.feature_b_name,
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        [
                                            f"Analyzed across {relationship.shared_example_count:,} shared examples",
                                        ],
                                        className="mb-0 small",
                                    ),
                                ],
                                color="info",
                                className="mb-0",
                            ),
                        ],
                        width=10,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dcc.Clipboard(
                                        target_id="relationship-markdown-content",
                                        title="Copy to clipboard (Markdown)",
                                        className="btn btn-outline-secondary btn-sm",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Small(
                                        "Copy MD",
                                        className="d-block text-muted mt-1",
                                    ),
                                ],
                                className="text-center",
                            ),
                        ],
                        width=2,
                        className="d-flex align-items-center justify-content-center",
                    ),
                ],
                className="mb-3",
            ),
            # Hidden div with markdown content for clipboard
            html.Div(
                markdown_text,
                id="relationship-markdown-content",
                style={"display": "none"},
            ),
            # Stats row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H3(
                                            f"{relationship.activation_correlation:.2f}",
                                            className="text-center mb-1",
                                        ),
                                        html.P(
                                            "Activation Correlation",
                                            className="text-muted text-center small mb-1",
                                        ),
                                        html.P(
                                            corr_interpretation,
                                            className="text-center small mb-0",
                                        ),
                                    ]
                                ),
                                className="h-100",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H3(
                                            f"{conflict_rate:.0f}%",
                                            className="text-center mb-1",
                                        ),
                                        html.P(
                                            "Conflict Rate",
                                            className="text-muted text-center small mb-1",
                                        ),
                                        html.P(
                                            f"{relationship.conflict_count} conflicts / {relationship.agreement_count} agreements",
                                            className="text-center small mb-0",
                                        ),
                                    ]
                                ),
                                className="h-100",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H3(
                                            f"{total_tokens}",
                                            className="text-center mb-1",
                                        ),
                                        html.P(
                                            "Tokens Influenced",
                                            className="text-muted text-center small mb-1",
                                        ),
                                        html.P(
                                            f"{relationship.tokens_both_influence} shared, {relationship.tokens_only_a}+{relationship.tokens_only_b} unique",
                                            className="text-center small mb-0",
                                        ),
                                    ]
                                ),
                                className="h-100",
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            # Interpretation
            dbc.Alert(conflict_interpretation, color="light", className="mb-3"),
            # Charts row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=build_activation_scatter(relationship),
                                config={"displayModeBar": False},
                            ),
                        ],
                        width=5,
                    ),
                    dbc.Col(
                        [
                            html.H6(
                                "Token Influence Summary", className="mb-2"
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        f"✓ Agree: {relationship.agreement_count}",
                                        className="badge bg-success me-2",
                                    ),
                                    html.Span(
                                        f"⚡ Conflict: {relationship.conflict_count}",
                                        className="badge bg-danger me-2",
                                    ),
                                    html.Span(
                                        f"— Neutral: {relationship.neutral_count}",
                                        className="badge bg-secondary",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            build_differentiators_section(relationship),
                        ],
                        width=7,
                    ),
                ],
                className="mb-3",
            ),
            # Full influence chart
            html.Hr(),
            dcc.Graph(
                figure=build_aggregate_influence_chart(relationship),
                config={"displayModeBar": False},
            ),
        ]
    )


def compute_percentile_range_data(
    feature_id: int,
    feature_name: str,
    percentile_low: float,
    percentile_high: float,
    inv_vocab: dict[str, str],
    weapon_names: dict[int, str],
) -> Optional[PercentileRangeData]:
    """
    Load examples in the given percentile range and compute distributions.

    Uses raw Zarr data loading to get ALL non-zero activations for accurate
    percentile calculation, not just the top examples in optimized storage.

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
    from collections import Counter

    import numpy as np
    import polars as pl

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if DASHBOARD_CONTEXT is None or DASHBOARD_CONTEXT.db is None:
        return None

    db = DASHBOARD_CONTEXT.db

    # Load ALL non-zero activations using raw data loading
    # This bypasses optimized storage to get accurate percentiles
    if hasattr(db, "_load_feature_activations"):
        # Use EfficientFSDatabase's raw loading
        _, all_activations = db._load_feature_activations(feature_id)
    else:
        # Fallback to regular method
        all_activations = db.get_feature_activations(feature_id, limit=None)

    if all_activations is None or len(all_activations) == 0:
        return None

    n = len(all_activations)
    if n == 0:
        return None

    # Get activation values to compute value-range percentiles
    act_col = "activation"
    activation_values = all_activations[act_col].to_numpy()

    # Compute percentile thresholds based on ACTIVATION VALUE RANGE
    # e.g., if activations range 0.0-1.0, top 10% = activations in 0.9-1.0
    # Formula: threshold = min + (percentile/100) * (max - min)
    min_act = float(activation_values.min())
    max_act = float(activation_values.max())
    act_range = max_act - min_act

    low_threshold = min_act + (percentile_low / 100) * act_range
    high_threshold = min_act + (percentile_high / 100) * act_range

    # Filter to examples within the activation value range
    range_df = all_activations.filter(
        (pl.col(act_col) >= low_threshold) & (pl.col(act_col) <= high_threshold)
    )

    if range_df.is_empty():
        return None

    # Sort by activation descending for consistent ordering
    range_df = range_df.sort(act_col, descending=True)

    # Get column names (handle both naming conventions)
    act_col = "activation"
    weapon_col = (
        "weapon_id" if "weapon_id" in range_df.columns else "weapon_id_token"
    )
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
        token_name = inv_vocab.get(token_id) or inv_vocab.get(
            str(token_id), str(token_id)
        )
        percentage = (count / example_count) * 100
        token_distribution[token_name] = (count, percentage)

    # Count weapon frequencies
    weapon_counts = Counter()
    for row in range_df.iter_rows(named=True):
        weapon_id = row.get(weapon_col)
        if weapon_id is not None:
            weapon_counts[weapon_id] += 1

    # Get weapon properties for hover info
    from splatnlp.dashboard.utils.converters import get_weapon_properties_df

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

    # Sample examples
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
) -> html.Div:
    """Build UI card for a single percentile range."""
    # Token list
    token_items = []
    for token_name, (count, pct) in list(range_data.token_distribution.items())[
        :8
    ]:
        token_items.append(
            html.Li(f"{token_name} ({pct:.0f}%)", className="small")
        )

    # Weapon list with sub/special hover
    weapon_items = []
    for weapon_name, (count, pct) in list(
        range_data.weapon_distribution.items()
    )[:5]:
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
        abilities_str = (
            ", ".join(ex["abilities"][:3]) if ex["abilities"] else "—"
        )
        example_items.append(
            html.Li(
                f"{ex['weapon']} - {abilities_str} (act: {ex['activation']:.2f})",
                className="small",
            )
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="mb-2"),
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
        className="h-100",
    )


def build_percentile_comparison_content(
    comparison: PercentileComparison,
) -> html.Div:
    """Build full modal content with side-by-side comparison."""
    # Generate markdown for clipboard
    markdown_text = format_percentile_markdown(comparison)

    return html.Div(
        [
            # Header with copy button
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.H5(
                                        [
                                            comparison.feature_a_name,
                                            html.Span(
                                                " vs ",
                                                className="text-muted mx-2",
                                            ),
                                            comparison.feature_b_name,
                                        ],
                                        className="mb-2",
                                    ),
                                    html.P(
                                        "Testing slider-scale hypothesis: comparing top vs middle percentile ranges",
                                        className="mb-0 small",
                                    ),
                                ],
                                color="info",
                                className="mb-0",
                            ),
                        ],
                        width=10,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dcc.Clipboard(
                                        target_id="percentile-markdown-content",
                                        title="Copy to clipboard (Markdown)",
                                        className="btn btn-outline-secondary btn-sm",
                                        style={"fontSize": "1.2rem"},
                                    ),
                                    html.Small(
                                        "Copy MD",
                                        className="d-block text-muted mt-1",
                                    ),
                                ],
                                className="text-center",
                            ),
                        ],
                        width=2,
                        className="d-flex align-items-center justify-content-center",
                    ),
                ],
                className="mb-3",
            ),
            # Hidden div with markdown content for clipboard
            html.Div(
                markdown_text,
                id="percentile-markdown-content",
                style={"display": "none"},
            ),
            # TOP 10% section
            html.H5("Top 10% (90th–100th percentile)", className="mb-3 mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        build_percentile_range_card(
                            comparison.top_range_a,
                            comparison.feature_a_name,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        build_percentile_range_card(
                            comparison.top_range_b,
                            comparison.feature_b_name,
                        ),
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # MIDDLE 20% section
            html.H5("Middle 20% (40th–60th percentile)", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        build_percentile_range_card(
                            comparison.middle_range_a,
                            comparison.feature_a_name,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        build_percentile_range_card(
                            comparison.middle_range_b,
                            comparison.feature_b_name,
                        ),
                        width=6,
                    ),
                ],
            ),
        ]
    )


def format_percentile_markdown(comparison: PercentileComparison) -> str:
    """Format percentile comparison as Markdown for Obsidian export."""

    def format_range(data: PercentileRangeData, label: str) -> list[str]:
        lines = [
            f"### {label}",
            "",
            f"**Activation range**: {data.activation_min:.2f} – {data.activation_max:.2f}",
            f"**Examples**: {data.example_count:,}",
            f"**Median activation**: {data.median_activation:.3f}",
            "",
            "**Top Tokens**:",
        ]
        for token_name, (count, pct) in list(data.token_distribution.items())[
            :8
        ]:
            lines.append(f"- `{token_name}` ({pct:.0f}%)")
        lines.append("")
        lines.append("**Top Weapons**:")
        for weapon_name, (count, pct) in list(data.weapon_distribution.items())[
            :5
        ]:
            lines.append(f"- {weapon_name} ({pct:.0f}%)")
        lines.append("")
        lines.append("**Net Output Influence**:")
        for inf in data.net_influences[:8]:
            sign = "+" if inf["net"] > 0 else ""
            lines.append(f"- `{inf['token']}`: {sign}{inf['net']:.3f}")
        lines.append("")
        lines.append("**Sample Builds**:")
        for ex in data.sample_examples[:3]:
            abilities = (
                ", ".join(ex["abilities"][:3]) if ex["abilities"] else "—"
            )
            lines.append(
                f"- {ex['weapon']} — {abilities} (act: {ex['activation']:.2f})"
            )
        lines.append("")
        return lines

    lines = [
        f"## Percentile Analysis: {comparison.feature_a_name} vs {comparison.feature_b_name}",
        "",
        "Testing slider-scale hypothesis by comparing top vs middle percentile ranges.",
        "",
        "---",
        "",
        f"## {comparison.feature_a_name}",
        "",
    ]
    lines.extend(format_range(comparison.top_range_a, "Top 10% (90th–100th)"))
    lines.extend(
        format_range(comparison.middle_range_a, "Middle 20% (40th–60th)")
    )

    lines.extend(
        [
            "---",
            "",
            f"## {comparison.feature_b_name}",
            "",
        ]
    )
    lines.extend(format_range(comparison.top_range_b, "Top 10% (90th–100th)"))
    lines.extend(
        format_range(comparison.middle_range_b, "Middle 20% (40th–60th)")
    )

    return "\n".join(lines)


def get_feature_comparison_data(
    feature_id: int,
    db: Any,
    analyzer: TFIDFAnalyzer,
    labels_manager: Any,
    limit: int = DEFAULT_ACTIVATION_LIMIT,
) -> Optional[ComparisonFeatureData]:
    """Load TFIDFAnalysis and activations for a feature."""
    try:
        activations_df = db.get_feature_activations(feature_id, limit=limit)
        if activations_df is None or activations_df.is_empty():
            return None

        # Compute TF-IDF analysis
        tfidf_analysis = analyzer.analyze(
            activations_df, labels_manager, feature_id
        )

        # Get example indices
        if "index" in activations_df.columns:
            example_indices = set(activations_df["index"].to_list())
        elif "global_index" in activations_df.columns:
            example_indices = set(activations_df["global_index"].to_list())
        else:
            example_indices = set()

        # Get display name
        display_name = f"Feature {feature_id}"
        if labels_manager:
            display_name = labels_manager.get_display_name(feature_id)

        return ComparisonFeatureData(
            feature_id=feature_id,
            display_name=display_name,
            tfidf_analysis=tfidf_analysis,
            activations_df=activations_df,
            example_indices=example_indices,
        )
    except Exception as e:
        logger.error(
            f"Error loading comparison data for feature {feature_id}: {e}"
        )
        return None


def compute_overlap_analysis(
    features_data: list[ComparisonFeatureData],
) -> OverlapAnalysis:
    """Compute overlap between multiple features' activating examples."""
    if not features_data:
        return OverlapAnalysis(
            shared_indices=set(),
            unique_indices={},
            shared_count=0,
            feature_counts={},
        )

    # Start with all indices from first feature
    shared = features_data[0].example_indices.copy()

    # Intersect with all other features
    for fd in features_data[1:]:
        shared = shared.intersection(fd.example_indices)

    # Compute unique indices per feature
    unique_indices = {}
    feature_counts = {}
    for fd in features_data:
        other_indices = set()
        for other_fd in features_data:
            if other_fd.feature_id != fd.feature_id:
                other_indices = other_indices.union(other_fd.example_indices)
        unique_indices[fd.feature_id] = fd.example_indices - other_indices
        feature_counts[fd.feature_id] = len(fd.example_indices)

    return OverlapAnalysis(
        shared_indices=shared,
        unique_indices=unique_indices,
        shared_count=len(shared),
        feature_counts=feature_counts,
    )


def build_comparison_card(feature_data: ComparisonFeatureData) -> dbc.Card:
    """Build a compact comparison card for a single feature."""
    if not feature_data.tfidf_analysis:
        return dbc.Card(
            dbc.CardBody(
                html.P(
                    f"No analysis available for {feature_data.display_name}",
                    className="text-muted",
                )
            )
        )

    analysis = feature_data.tfidf_analysis

    # TF-IDF tokens
    tfidf_badges = []
    for token in analysis.top_tokens[:5]:  # Show top 5
        tfidf_badges.append(
            dbc.Badge(
                f"{token['ability_name']} ({token['tf_idf']:.2f})",
                color="primary",
                className="me-1 mb-1",
                pill=True,
            )
        )

    # Top weapons
    weapon_items = []
    for weapon in analysis.top_weapons[:3]:  # Show top 3
        weapon_items.append(
            html.Div(
                [
                    html.Span(weapon["weapon_name"], className="me-2"),
                    html.Small(
                        f"({weapon['percentage']:.0%})",
                        className="text-muted",
                    ),
                ],
                className="mb-1",
            )
        )

    # Class stats
    class_items = []
    for cls in analysis.class_stats[:3]:
        class_items.append(
            dbc.Badge(
                f"{cls['class']} ({cls['percentage']:.0%})",
                color="secondary",
                className="me-1",
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H6(
                    feature_data.display_name,
                    className="mb-0 text-truncate",
                    title=feature_data.display_name,
                )
            ),
            dbc.CardBody(
                [
                    # Example count
                    html.Div(
                        [
                            html.Small("Examples: ", className="text-muted"),
                            html.Span(
                                f"{len(feature_data.example_indices):,}",
                                className="fw-bold",
                            ),
                        ],
                        className="mb-2",
                    ),
                    # TF-IDF tokens
                    html.Div(
                        [
                            html.Small(
                                "Top Tokens:", className="text-muted d-block"
                            ),
                            html.Div(
                                tfidf_badges, className="d-flex flex-wrap"
                            ),
                        ],
                        className="mb-2",
                    ),
                    # Top weapons
                    html.Div(
                        [
                            html.Small(
                                "Top Weapons:", className="text-muted d-block"
                            ),
                            html.Div(weapon_items),
                        ],
                        className="mb-2",
                    ),
                    # Classes
                    html.Div(
                        [
                            html.Small(
                                "Classes:", className="text-muted d-block"
                            ),
                            html.Div(class_items, className="d-flex flex-wrap"),
                        ],
                    ),
                ],
                style={"fontSize": "0.85rem"},
            ),
        ],
        className="h-100",
    )


def get_feature_influence_data(feature_id: int) -> Optional[dict]:
    """Get influence data for a feature from the dashboard context."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "influence_data")
        or DASHBOARD_CONTEXT.influence_data is None
    ):
        return None

    influence_df = DASHBOARD_CONTEXT.influence_data
    feature_data = influence_df[influence_df["feature_id"] == feature_id]

    if feature_data.empty:
        return None

    feature_row = feature_data.iloc[0]

    # Extract positive and negative influences
    pos_data = []
    neg_data = []

    for i in range(1, 31):  # Top 30
        pos_tok_col = f"+{i}_tok"
        pos_val_col = f"+{i}_val"
        neg_tok_col = f"-{i}_tok"
        neg_val_col = f"-{i}_val"

        if pos_tok_col in feature_row and pd.notna(feature_row[pos_tok_col]):
            pos_data.append(
                {
                    "token": feature_row[pos_tok_col],
                    "value": float(feature_row[pos_val_col]),
                }
            )

        if neg_tok_col in feature_row and pd.notna(feature_row[neg_tok_col]):
            neg_data.append(
                {
                    "token": feature_row[neg_tok_col],
                    "value": float(feature_row[neg_val_col]),
                }
            )

    return {
        "positive": pos_data,
        "negative": neg_data,
    }


def build_logit_comparison(
    left_name: str,
    right_name: str,
    left_influence: Optional[dict],
    right_influence: Optional[dict],
) -> html.Div:
    """Build a side-by-side comparison of logit influences for two features."""
    if not left_influence and not right_influence:
        return html.Div(
            "Influence data not available for these features.",
            className="text-muted",
        )

    # Build comparison tables showing tokens influenced by each feature
    def build_influence_table(
        influence_data: Optional[dict],
        feature_name: str,
        influence_type: str,  # "positive" or "negative"
    ) -> html.Div:
        if not influence_data or not influence_data.get(influence_type):
            return html.Div(
                f"No {influence_type} influence data",
                className="text-muted small",
            )

        items = influence_data[influence_type][:15]  # Top 15
        color_class = (
            "text-success" if influence_type == "positive" else "text-danger"
        )

        rows = []
        for item in items:
            rows.append(
                html.Tr(
                    [
                        html.Td(item["token"], className="small"),
                        html.Td(
                            f"{item['value']:.4f}",
                            className=f"small text-end {color_class}",
                        ),
                    ]
                )
            )

        return html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Token", className="small"),
                            html.Th("Influence", className="small text-end"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            className="table table-sm table-striped mb-0",
        )

    # Find common tokens between the two features
    def find_common_tokens(
        left: Optional[dict], right: Optional[dict], influence_type: str
    ) -> set:
        if not left or not right:
            return set()
        left_tokens = {item["token"] for item in left.get(influence_type, [])}
        right_tokens = {item["token"] for item in right.get(influence_type, [])}
        return left_tokens.intersection(right_tokens)

    common_positive = find_common_tokens(
        left_influence, right_influence, "positive"
    )
    common_negative = find_common_tokens(
        left_influence, right_influence, "negative"
    )

    # Truncate names for display
    left_short = (left_name[:15] + "…") if len(left_name) > 15 else left_name
    right_short = (
        (right_name[:15] + "…") if len(right_name) > 15 else right_name
    )

    return html.Div(
        [
            # Positive influences comparison
            html.H6(
                "Positive Influences (tokens more likely)",
                className="text-success mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                left_short,
                                className="fw-bold small mb-1",
                                title=left_name,
                            ),
                            html.Div(
                                build_influence_table(
                                    left_influence, left_name, "positive"
                                ),
                                style={
                                    "maxHeight": "250px",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.P(
                                right_short,
                                className="fw-bold small mb-1",
                                title=right_name,
                            ),
                            html.Div(
                                build_influence_table(
                                    right_influence, right_name, "positive"
                                ),
                                style={
                                    "maxHeight": "250px",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            # Common positive tokens
            (
                html.P(
                    [
                        html.Strong("Common positive tokens: "),
                        (
                            ", ".join(sorted(common_positive)[:10])
                            if common_positive
                            else "None"
                        ),
                        (
                            f" (+{len(common_positive) - 10} more)"
                            if len(common_positive) > 10
                            else ""
                        ),
                    ],
                    className="text-muted small mb-3",
                )
                if common_positive
                else None
            ),
            html.Hr(),
            # Negative influences comparison
            html.H6(
                "Negative Influences (tokens less likely)",
                className="text-danger mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                left_short,
                                className="fw-bold small mb-1",
                                title=left_name,
                            ),
                            html.Div(
                                build_influence_table(
                                    left_influence, left_name, "negative"
                                ),
                                style={
                                    "maxHeight": "250px",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.P(
                                right_short,
                                className="fw-bold small mb-1",
                                title=right_name,
                            ),
                            html.Div(
                                build_influence_table(
                                    right_influence, right_name, "negative"
                                ),
                                style={
                                    "maxHeight": "250px",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            # Common negative tokens
            (
                html.P(
                    [
                        html.Strong("Common negative tokens: "),
                        (
                            ", ".join(sorted(common_negative)[:10])
                            if common_negative
                            else "None"
                        ),
                        (
                            f" (+{len(common_negative) - 10} more)"
                            if len(common_negative) > 10
                            else ""
                        ),
                    ],
                    className="text-muted small mt-2",
                )
                if common_negative
                else None
            ),
        ]
    )


def build_overlap_matrix(
    features_data: list[ComparisonFeatureData],
) -> go.Figure:
    """Build an n×n heatmap showing pairwise overlap counts between features."""
    import numpy as np

    n = len(features_data)
    matrix = np.zeros((n, n), dtype=int)
    labels = [fd.display_name for fd in features_data]
    # Truncate long labels for display
    short_labels = [
        (lbl[:20] + "...") if len(lbl) > 23 else lbl for lbl in labels
    ]

    # Compute pairwise overlaps
    for i, fd_i in enumerate(features_data):
        for j, fd_j in enumerate(features_data):
            if i == j:
                # Diagonal: total examples for this feature
                matrix[i, j] = len(fd_i.example_indices)
            else:
                # Off-diagonal: shared examples
                shared = fd_i.example_indices.intersection(fd_j.example_indices)
                matrix[i, j] = len(shared)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=short_labels,
            y=short_labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate=(
                "Feature A: %{y}<br>"
                "Feature B: %{x}<br>"
                "Shared: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Pairwise Overlap Matrix (Top 100 Examples)",
        xaxis_title="",
        yaxis_title="",
        height=max(300, 80 * n),  # Scale height with number of features
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickangle=45),
    )

    return fig


def build_overlap_summary_table(
    features_data: list[ComparisonFeatureData],
    overlap: OverlapAnalysis,
) -> html.Div:
    """Build a summary table for 3+ feature overlap."""
    rows = []
    for fd in features_data:
        unique_count = len(overlap.unique_indices.get(fd.feature_id, set()))
        total = overlap.feature_counts.get(fd.feature_id, 0)
        rows.append(
            html.Tr(
                [
                    html.Td(fd.display_name),
                    html.Td(f"{total:,}"),
                    html.Td(f"{unique_count:,}"),
                    html.Td(f"{overlap.shared_count:,}"),
                ]
            )
        )

    return html.Div(
        [
            html.H6("Overlap Summary (All Features)", className="mb-2"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Feature"),
                                html.Th("Total Examples"),
                                html.Th("Unique"),
                                html.Th("Shared (all)"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="table table-sm table-striped",
            ),
        ]
    )


def build_shared_examples_table(
    shared_indices: set[int],
    left_name: str,
    right_name: str,
    left_examples: dict[int, dict],
    right_examples: dict[int, dict],
    weapon_names: dict[int, str],
    inv_vocab: dict[str, str],
    max_examples: int = 20,
) -> html.Div:
    """Build a table showing shared examples with weapon, tokens, and activations."""
    if not shared_indices:
        return html.Div(
            "No shared examples found.",
            className="text-muted small",
        )

    # Calculate max activations for context
    left_max = max(
        (ex.get("activation", 0.0) for ex in left_examples.values()),
        default=0.0,
    )
    right_max = max(
        (ex.get("activation", 0.0) for ex in right_examples.values()),
        default=0.0,
    )

    # Build rows sorted by sum of activations (most activated first)
    rows_data = []
    for idx in shared_indices:
        left_data = left_examples.get(idx, {})
        right_data = right_examples.get(idx, {})
        left_act = left_data.get("activation", 0.0)
        right_act = right_data.get("activation", 0.0)

        # Get weapon name - try both examples in case one is missing
        weapon_id = left_data.get("weapon_id") or right_data.get("weapon_id")
        # Handle both int and string keys (JSON serialization converts to strings)
        weapon_name = (
            (
                weapon_names.get(weapon_id)
                or weapon_names.get(str(weapon_id))
                or f"Weapon {weapon_id}"
            )
            if weapon_id
            else "Unknown"
        )

        # Get ability tokens - use left or right
        ability_tokens = (
            left_data.get("ability_tokens")
            or right_data.get("ability_tokens")
            or []
        )
        ability_names = [
            inv_vocab.get(str(t), str(t)) for t in ability_tokens[:5]
        ]  # Limit to 5

        rows_data.append(
            {
                "index": idx,
                "weapon_name": weapon_name,
                "abilities": ", ".join(ability_names) if ability_names else "—",
                "left_act": left_act,
                "right_act": right_act,
                "sum_act": left_act + right_act,
            }
        )

    # Sort by sum of activations descending
    rows_data.sort(key=lambda x: x["sum_act"], reverse=True)

    # Truncate long names for headers
    left_short = (left_name[:12] + "…") if len(left_name) > 12 else left_name
    right_short = (
        (right_name[:12] + "…") if len(right_name) > 12 else right_name
    )

    # Build table rows - clickable for influence analysis
    table_rows = []
    for row in rows_data[:max_examples]:
        table_rows.append(
            html.Tr(
                [
                    html.Td(row["weapon_name"], className="small"),
                    html.Td(
                        row["abilities"],
                        className="small",
                        style={
                            "maxWidth": "200px",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                    ),
                    html.Td(
                        f"{row['left_act']:.3f}", className="small text-end"
                    ),
                    html.Td(
                        f"{row['right_act']:.3f}", className="small text-end"
                    ),
                ],
                id={"type": "shared-example-row", "index": row["index"]},
                n_clicks=0,
                style={"cursor": "pointer"},
                className="hover-highlight",
            )
        )

    return html.Div(
        [
            html.H6(
                f"Shared Examples ({len(shared_indices)} total)",
                className="mt-3 mb-2",
            ),
            # Show max activation context
            html.P(
                [
                    html.Span("Max activations: ", className="fw-bold"),
                    html.Span(f"{left_short}: {left_max:.3f}", title=left_name),
                    html.Span(" | "),
                    html.Span(
                        f"{right_short}: {right_max:.3f}", title=right_name
                    ),
                ],
                className="text-muted small mb-2",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Weapon", className="small"),
                                    html.Th("Abilities", className="small"),
                                    html.Th(
                                        left_short,
                                        className="small text-end",
                                        title=left_name,
                                    ),
                                    html.Th(
                                        right_short,
                                        className="small text-end",
                                        title=right_name,
                                    ),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    className="table table-sm table-striped",
                ),
                style={"maxHeight": "300px", "overflowY": "auto"},
            ),
            (
                html.P(
                    f"Showing top {min(max_examples, len(shared_indices))} by combined activation",
                    className="text-muted small mb-0",
                )
                if len(shared_indices) > max_examples
                else None
            ),
            html.P(
                "Click a row to analyze activation-weighted influence",
                className="text-muted small fst-italic mb-0 mt-1",
            ),
        ]
    )


def build_head_to_head_ui(
    features_data: list[ComparisonFeatureData],
    weapon_names: dict[int, str],
    inv_vocab: dict[str, str],
) -> html.Div:
    """Build UI for head-to-head feature comparison with logit influences."""
    options = [
        {"label": fd.display_name, "value": fd.feature_id}
        for fd in features_data
    ]

    # Default to first two features
    default_left = features_data[0] if len(features_data) > 0 else None
    default_right = features_data[1] if len(features_data) > 1 else None

    # Build initial content for the default selection
    initial_content = []
    if default_left and default_right:
        # Compute overlap for shared examples
        left_indices = default_left.example_indices
        right_indices = default_right.example_indices
        shared = left_indices.intersection(right_indices)

        initial_overlap = OverlapAnalysis(
            shared_indices=shared,
            unique_indices={
                default_left.feature_id: left_indices - right_indices,
                default_right.feature_id: right_indices - left_indices,
            },
            shared_count=len(shared),
            feature_counts={
                default_left.feature_id: len(left_indices),
                default_right.feature_id: len(right_indices),
            },
        )

        # Build example maps for shared examples table
        def get_examples_map(fd: ComparisonFeatureData) -> dict[int, dict]:
            examples_map = {}
            index_col = (
                "index"
                if "index" in fd.activations_df.columns
                else "global_index"
            )
            weapon_col = (
                "weapon_id"
                if "weapon_id" in fd.activations_df.columns
                else "weapon_id_token"
            )
            ability_col = (
                "ability_tokens"
                if "ability_tokens" in fd.activations_df.columns
                else "ability_input_tokens"
            )
            for row in fd.activations_df.iter_rows(named=True):
                idx = row.get(index_col)
                if idx is not None:
                    examples_map[idx] = {
                        "weapon_id": row.get(weapon_col),
                        "ability_tokens": row.get(ability_col, []),
                        "activation": row.get("activation", 0.0),
                    }
            return examples_map

        left_examples = get_examples_map(default_left)
        right_examples = get_examples_map(default_right)

        # Get logit influence data
        left_influence = get_feature_influence_data(default_left.feature_id)
        right_influence = get_feature_influence_data(default_right.feature_id)

        # Build logit comparison
        logit_comparison = build_logit_comparison(
            default_left.display_name,
            default_right.display_name,
            left_influence,
            right_influence,
        )

        # Build shared examples table
        shared_table = build_shared_examples_table(
            shared,
            default_left.display_name,
            default_right.display_name,
            left_examples,
            right_examples,
            weapon_names,
            inv_vocab,
        )

        initial_content = [
            # Logit comparison section
            logit_comparison,
            html.Hr(className="my-3"),
            # Overlap summary
            html.P(
                [
                    html.Strong("Example Overlap: "),
                    f"{initial_overlap.shared_count:,} shared examples ",
                    f"({initial_overlap.shared_count / max(1, min(len(left_indices), len(right_indices))) * 100:.1f}% of smaller set)",
                ],
                className="text-muted small mb-2",
            ),
            shared_table,
        ]

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Head-to-Head Comparison", className="mb-3"),
                html.P(
                    "Select two features to compare output logits and shared examples:",
                    className="text-muted small mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Feature A:", className="small"),
                                dcc.Dropdown(
                                    id="venn-feature-left",
                                    options=options,
                                    value=(
                                        default_left.feature_id
                                        if default_left
                                        else None
                                    ),
                                    clearable=False,
                                    className="mb-2",
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        "vs", className="text-center fw-bold"
                                    ),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Relationship",
                                                id="analyze-relationship-btn",
                                                color="primary",
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Percentiles",
                                                id="analyze-percentile-btn",
                                                color="secondary",
                                                size="sm",
                                            ),
                                        ],
                                        className="mt-2",
                                    ),
                                ],
                                className="text-center mt-3",
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Feature B:", className="small"),
                                dcc.Dropdown(
                                    id="venn-feature-right",
                                    options=options,
                                    value=(
                                        default_right.feature_id
                                        if default_right
                                        else None
                                    ),
                                    clearable=False,
                                    className="mb-2",
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-3",
                ),
                # Head-to-head content - show initial render, updated by callback
                html.Div(id="venn-diagram-container", children=initial_content),
            ]
        ),
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("primary-feature-display", "children"),
    Input("feature-dropdown", "value"),
)
def update_primary_feature_display(selected_feature_id: Optional[int]):
    """Display the currently selected primary feature."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return "No feature selected"

    labels_manager = getattr(DASHBOARD_CONTEXT, "feature_labels_manager", None)
    if labels_manager:
        return labels_manager.get_display_name(selected_feature_id)
    return f"Feature {selected_feature_id}"


@callback(
    Output("comparison-features-dropdown", "options"),
    [
        Input("feature-dropdown", "value"),
        Input("feature-labels-updated", "data"),
    ],
)
def update_comparison_dropdown_options(
    selected_feature_id: Optional[int],
    _labels_updated: Optional[int],
):
    """Populate comparison dropdown with all features except the selected one."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not hasattr(DASHBOARD_CONTEXT, "db") or DASHBOARD_CONTEXT.db is None:
        return []

    db = DASHBOARD_CONTEXT.db
    labels_manager = getattr(DASHBOARD_CONTEXT, "feature_labels_manager", None)

    try:
        feature_ids = db.get_all_feature_ids()
        options = []
        for fid in feature_ids:
            if fid == selected_feature_id:
                continue  # Skip the primary feature
            if labels_manager:
                label = labels_manager.get_display_name(fid)
            else:
                label = f"Feature {fid}"
            options.append({"label": label, "value": fid})
        return options
    except Exception as e:
        logger.error(f"Error loading feature options: {e}")
        return []


@callback(
    [
        Output("comparison-cards-container", "children"),
        Output("overlap-matrix-container", "children"),
        Output("overlap-analysis-container", "children"),
        Output("venn-subselect-container", "children"),
        Output("comparison-data-cache", "data"),
        Output("comparison-error-message", "children"),
    ],
    [
        Input("feature-dropdown", "value"),
        Input("comparison-features-dropdown", "value"),
        Input("active-tab-store", "data"),
    ],
)
def render_comparison(
    primary_feature_id: Optional[int],
    comparison_feature_ids: Optional[list[int]],
    active_tab: Optional[str],
):
    """Render the feature comparison view."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Lazy loading: skip if tab is not active
    if active_tab != "tab-comparison":
        return no_update, no_update, no_update, no_update, no_update, no_update

    if primary_feature_id is None:
        return [], [], [], [], None, "Select a primary feature to compare."

    if not comparison_feature_ids:
        return (
            [
                dbc.Alert(
                    "Select one or more features from the dropdown above to compare.",
                    color="info",
                )
            ],
            [],
            [],
            [],
            None,
            "",
        )

    if not hasattr(DASHBOARD_CONTEXT, "db") or DASHBOARD_CONTEXT.db is None:
        return [], [], [], [], None, "Database not available."

    try:
        db = DASHBOARD_CONTEXT.db
        labels_manager = getattr(
            DASHBOARD_CONTEXT, "feature_labels_manager", None
        )

        # Create analyzer (reuse pattern from intervals_grid)
        from splatnlp.dashboard.components.intervals_grid_component import (
            TFIDFAnalyzer,
        )
        from splatnlp.preprocessing.transform.mappings import generate_maps

        analyzer = TFIDFAnalyzer(
            DASHBOARD_CONTEXT.inv_vocab,
            DASHBOARD_CONTEXT.inv_weapon_vocab,
            generate_maps()[1],
            db.idf,
        )

        # Load data for all features
        all_feature_ids = [primary_feature_id] + list(comparison_feature_ids)[
            : MAX_COMPARISON_FEATURES - 1
        ]
        features_data = []

        for fid in all_feature_ids:
            fd = get_feature_comparison_data(fid, db, analyzer, labels_manager)
            if fd:
                features_data.append(fd)

        if len(features_data) < 2:
            return (
                [],
                [],
                [],
                [],
                None,
                "Not enough feature data available for comparison.",
            )

        # Build comparison cards with row wrapping (max 4 per row)
        cards_per_row = min(len(features_data), MAX_CARDS_PER_ROW)
        col_width = 12 // cards_per_row

        card_rows = []
        current_row = []
        for i, fd in enumerate(features_data):
            current_row.append(
                dbc.Col(
                    build_comparison_card(fd), width=col_width, className="mb-3"
                )
            )
            # Start new row after MAX_CARDS_PER_ROW or at the end
            if (
                len(current_row) >= MAX_CARDS_PER_ROW
                or i == len(features_data) - 1
            ):
                card_rows.append(dbc.Row(current_row))
                current_row = []

        cards_container = html.Div(card_rows)

        # Build overlap matrix (n×n heatmap)
        matrix_fig = build_overlap_matrix(features_data)
        matrix_content = dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(
                        figure=matrix_fig, config={"displayModeBar": False}
                    ),
                ]
            ),
            className="mb-3",
        )

        # Build overlap analysis
        overlap = compute_overlap_analysis(features_data)

        # Cache feature data for Venn sub-selection (store as serializable dict)
        # Include full example data for showing shared examples with weapon/tokens
        cache_data = {
            "_weapon_names": {
                str(k): v for k, v in analyzer.id_to_name_mapping.items()
            },
            "_inv_vocab": DASHBOARD_CONTEXT.inv_vocab,
        }
        for fd in features_data:
            # Build index -> example data mapping (weapon_id, ability_tokens, activation)
            examples_map = {}
            index_col = (
                "index"
                if "index" in fd.activations_df.columns
                else "global_index"
            )
            weapon_col = (
                "weapon_id"
                if "weapon_id" in fd.activations_df.columns
                else "weapon_id_token"
            )
            ability_col = (
                "ability_tokens"
                if "ability_tokens" in fd.activations_df.columns
                else "ability_input_tokens"
            )

            for row in fd.activations_df.iter_rows(named=True):
                idx = row.get(index_col)
                if idx is not None:
                    examples_map[idx] = {
                        "weapon_id": row.get(weapon_col),
                        "ability_tokens": row.get(ability_col, []),
                        "activation": row.get("activation", 0.0),
                    }

            cache_data[fd.feature_id] = {
                "feature_id": fd.feature_id,
                "display_name": fd.display_name,
                "example_indices": list(fd.example_indices),
                "examples": examples_map,
            }

        if len(features_data) == 2:
            # Show overlap statistics and head-to-head UI for 2 features
            overlap_content = dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            [
                                html.Strong("Overlap Statistics: "),
                                f"{overlap.shared_count:,} examples appear in both features ",
                                f"({overlap.shared_count / max(1, min(overlap.feature_counts.values())) * 100:.1f}% of smaller set)",
                            ],
                            className="text-muted small mb-0",
                        ),
                    ]
                ),
                className="mb-3",
            )
            # Show head-to-head UI for 2 features as well
            venn_subselect = build_head_to_head_ui(
                features_data,
                analyzer.id_to_name_mapping,
                DASHBOARD_CONTEXT.inv_vocab,
            )
        else:
            # Summary table for 3+ features
            overlap_content = dbc.Card(
                dbc.CardBody(
                    [
                        build_overlap_summary_table(features_data, overlap),
                    ]
                ),
                className="mb-3",
            )
            # Show head-to-head comparison UI
            venn_subselect = build_head_to_head_ui(
                features_data,
                analyzer.id_to_name_mapping,
                DASHBOARD_CONTEXT.inv_vocab,
            )

        return (
            cards_container,
            matrix_content,
            overlap_content,
            venn_subselect,
            cache_data,
            "",
        )

    except Exception as e:
        logger.error(f"Error rendering comparison: {e}", exc_info=True)
        return [], [], [], [], None, f"Error: {str(e)}"


@callback(
    Output("venn-diagram-container", "children"),
    [
        Input("venn-feature-left", "value"),
        Input("venn-feature-right", "value"),
    ],
    State("comparison-data-cache", "data"),
    prevent_initial_call=True,
)
def render_head_to_head_comparison(
    left_feature_id: Optional[int],
    right_feature_id: Optional[int],
    cache_data: Optional[dict],
):
    """Render head-to-head comparison with logit influences and shared examples."""
    if not left_feature_id or not right_feature_id:
        return html.Div(
            "Select two features to compare.", className="text-muted"
        )

    if left_feature_id == right_feature_id:
        return dbc.Alert(
            "Please select two different features to compare.",
            color="warning",
        )

    if not cache_data:
        return html.Div("No comparison data available.", className="text-muted")

    # Get cached data for the selected features
    left_data = cache_data.get(str(left_feature_id)) or cache_data.get(
        left_feature_id
    )
    right_data = cache_data.get(str(right_feature_id)) or cache_data.get(
        right_feature_id
    )

    if not left_data or not right_data:
        return html.Div(
            "Feature data not found in cache.", className="text-muted"
        )

    # Get display names
    left_name = left_data["display_name"]
    right_name = right_data["display_name"]

    # Get logit influence data
    left_influence = get_feature_influence_data(left_feature_id)
    right_influence = get_feature_influence_data(right_feature_id)

    # Build logit comparison
    logit_comparison = build_logit_comparison(
        left_name,
        right_name,
        left_influence,
        right_influence,
    )

    # Compute overlap for shared examples
    left_indices = set(left_data["example_indices"])
    right_indices = set(right_data["example_indices"])
    shared = left_indices.intersection(right_indices)

    # Get examples data from cache for shared examples table
    left_examples = left_data.get("examples", {})
    right_examples = right_data.get("examples", {})

    # Convert string keys back to int if needed (JSON serialization)
    if left_examples and isinstance(next(iter(left_examples.keys())), str):
        left_examples = {int(k): v for k, v in left_examples.items()}
    if right_examples and isinstance(next(iter(right_examples.keys())), str):
        right_examples = {int(k): v for k, v in right_examples.items()}

    # Get weapon names and vocab from cache
    weapon_names = cache_data.get("_weapon_names", {})
    inv_vocab = cache_data.get("_inv_vocab", {})

    # Build shared examples table
    shared_table = build_shared_examples_table(
        shared,
        left_name,
        right_name,
        left_examples,
        right_examples,
        weapon_names,
        inv_vocab,
    )

    return html.Div(
        [
            logit_comparison,
            html.Hr(className="my-3"),
            html.P(
                [
                    html.Strong("Example Overlap: "),
                    f"{len(shared):,} shared examples ",
                    f"({len(shared) / max(1, min(len(left_indices), len(right_indices))) * 100:.1f}% of smaller set)",
                ],
                className="text-muted small mb-2",
            ),
            shared_table,
        ]
    )


@callback(
    Output("influence-analysis-modal", "is_open"),
    Output("influence-analysis-content", "children"),
    Input({"type": "shared-example-row", "index": ALL}, "n_clicks"),
    State("venn-feature-left", "value"),
    State("venn-feature-right", "value"),
    State("comparison-data-cache", "data"),
    prevent_initial_call=True,
)
def show_influence_analysis(
    n_clicks_list: list,
    feature_a_id: Optional[int],
    feature_b_id: Optional[int],
    cache_data: Optional[dict],
):
    """Show modal with influence breakdown when a shared example row is clicked."""
    # Check if any row was actually clicked
    if not n_clicks_list or not any(n_clicks_list):
        return no_update, no_update

    # Find which row was clicked using callback context
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update

    # Get the clicked row's index from the trigger
    triggered_id = ctx.triggered[0]["prop_id"]
    if "shared-example-row" not in triggered_id:
        return no_update, no_update

    # Parse the index from the triggered ID
    import json

    try:
        # Extract the JSON part from "{'type': 'shared-example-row', 'index': 123}.n_clicks"
        json_str = triggered_id.replace(".n_clicks", "")
        id_dict = json.loads(json_str)
        example_index = id_dict["index"]
    except (json.JSONDecodeError, KeyError):
        return no_update, no_update

    # Validate we have the required data
    if not feature_a_id or not feature_b_id or not cache_data:
        return True, html.Div(
            "Missing feature selection or cache data.", className="text-warning"
        )

    # Get cached data for the selected features
    left_data = cache_data.get(str(feature_a_id)) or cache_data.get(
        feature_a_id
    )
    right_data = cache_data.get(str(feature_b_id)) or cache_data.get(
        feature_b_id
    )

    if not left_data or not right_data:
        return True, html.Div(
            "Feature data not found in cache.", className="text-warning"
        )

    # Get example data and activation values
    left_examples = left_data.get("examples", {})
    right_examples = right_data.get("examples", {})

    # Convert string keys back to int if needed
    if left_examples and isinstance(next(iter(left_examples.keys())), str):
        left_examples = {int(k): v for k, v in left_examples.items()}
    if right_examples and isinstance(next(iter(right_examples.keys())), str):
        right_examples = {int(k): v for k, v in right_examples.items()}

    # Get activation values for this example
    left_ex = left_examples.get(example_index, {})
    right_ex = right_examples.get(example_index, {})
    activation_a = left_ex.get("activation", 0.0)
    activation_b = right_ex.get("activation", 0.0)

    # Get example details for display
    weapon_names = cache_data.get("_weapon_names", {})
    inv_vocab = cache_data.get("_inv_vocab", {})
    weapon_id = left_ex.get("weapon_id") or right_ex.get("weapon_id")
    weapon_name = (
        weapon_names.get(weapon_id)
        or weapon_names.get(str(weapon_id))
        or f"Weapon {weapon_id}"
    )
    ability_tokens = (
        left_ex.get("ability_tokens") or right_ex.get("ability_tokens") or []
    )
    ability_names = [inv_vocab.get(str(t), str(t)) for t in ability_tokens[:5]]

    # Compute influence breakdown
    breakdowns = compute_influence_breakdown(
        feature_a_id, feature_b_id, activation_a, activation_b
    )

    # Build the modal content
    feature_a_name = left_data["display_name"]
    feature_b_name = right_data["display_name"]

    content = html.Div(
        [
            # Example header
            html.Div(
                [
                    html.H5(f"{weapon_name}", className="mb-1"),
                    html.P(
                        (
                            ", ".join(ability_names)
                            if ability_names
                            else "No abilities"
                        ),
                        className="text-muted small mb-0",
                    ),
                ],
                className="mb-3 p-2 bg-light rounded",
            ),
            # Feature activation summary
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        feature_a_name, className="fw-bold"
                                    ),
                                    html.Br(),
                                    html.Span(
                                        f"Activation: {activation_a:.4f}",
                                        className="text-primary",
                                    ),
                                ],
                                className="p-2 border rounded",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        feature_b_name, className="fw-bold"
                                    ),
                                    html.Br(),
                                    html.Span(
                                        f"Activation: {activation_b:.4f}",
                                        className="text-danger",
                                    ),
                                ],
                                className="p-2 border rounded",
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # Explanation
            html.P(
                [
                    "Net influence = raw influence × activation. ",
                    "This shows how much each feature contributes to output token probabilities for this specific example.",
                ],
                className="text-muted small mb-3",
            ),
            # Bar chart
            dcc.Graph(
                figure=build_influence_comparison_chart(
                    breakdowns, feature_a_name, feature_b_name
                ),
                config={"displayModeBar": False},
            ),
            html.Hr(),
            # Breakdown table
            html.H6("Detailed Breakdown", className="mb-2"),
            build_influence_breakdown_table(
                breakdowns, feature_a_name, feature_b_name
            ),
        ]
    )

    return True, content


@callback(
    Output("relationship-analysis-modal", "is_open"),
    Output("relationship-analysis-content", "children"),
    Input("analyze-relationship-btn", "n_clicks"),
    State("venn-feature-left", "value"),
    State("venn-feature-right", "value"),
    State("comparison-data-cache", "data"),
    prevent_initial_call=True,
)
def show_relationship_analysis(
    n_clicks: int,
    feature_a_id: Optional[int],
    feature_b_id: Optional[int],
    cache_data: Optional[dict],
):
    """Show modal with aggregate feature relationship analysis."""
    if not n_clicks:
        return False, no_update

    if not feature_a_id or not feature_b_id:
        return True, dbc.Alert(
            "Please select two features to analyze.",
            color="warning",
        )

    if feature_a_id == feature_b_id:
        return True, dbc.Alert(
            "Please select two different features.",
            color="warning",
        )

    if not cache_data:
        return True, dbc.Alert(
            "No comparison data available. Please run a comparison first.",
            color="warning",
        )

    # Get cached data
    left_data = cache_data.get(str(feature_a_id)) or cache_data.get(
        feature_a_id
    )
    right_data = cache_data.get(str(feature_b_id)) or cache_data.get(
        feature_b_id
    )

    if not left_data or not right_data:
        return True, dbc.Alert(
            "Feature data not found in cache.",
            color="warning",
        )

    # Get examples from cache
    left_examples = left_data.get("examples", {})
    right_examples = right_data.get("examples", {})

    # Convert string keys back to int if needed (JSON serialization)
    if left_examples and isinstance(next(iter(left_examples.keys())), str):
        left_examples = {int(k): v for k, v in left_examples.items()}
    if right_examples and isinstance(next(iter(right_examples.keys())), str):
        right_examples = {int(k): v for k, v in right_examples.items()}

    # Compute shared indices
    left_indices = set(left_data.get("example_indices", []))
    right_indices = set(right_data.get("example_indices", []))
    shared_indices = left_indices & right_indices

    if not shared_indices:
        return True, dbc.Alert(
            "These features have no shared examples to analyze.",
            color="info",
        )

    # Compute relationship
    relationship = compute_feature_relationship(
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
        feature_a_name=left_data["display_name"],
        feature_b_name=right_data["display_name"],
        shared_indices=shared_indices,
        examples_a=left_examples,
        examples_b=right_examples,
    )

    # Build modal content
    content = build_relationship_analysis_content(relationship)

    return True, content


@callback(
    Output("percentile-analysis-modal", "is_open"),
    Output("percentile-analysis-content", "children"),
    Input("analyze-percentile-btn", "n_clicks"),
    State("venn-feature-left", "value"),
    State("venn-feature-right", "value"),
    State("comparison-data-cache", "data"),
    prevent_initial_call=True,
)
def show_percentile_analysis(
    n_clicks: int,
    feature_a_id: Optional[int],
    feature_b_id: Optional[int],
    cache_data: Optional[dict],
):
    """Show modal with percentile range comparison."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return False, no_update

    if not feature_a_id or not feature_b_id:
        return True, dbc.Alert(
            "Please select two features to analyze.",
            color="warning",
        )

    if feature_a_id == feature_b_id:
        return True, dbc.Alert(
            "Please select two different features.",
            color="warning",
        )

    if not cache_data:
        return True, dbc.Alert(
            "No comparison data available. Please run a comparison first.",
            color="warning",
        )

    # Get cached data for names
    left_data = cache_data.get(str(feature_a_id)) or cache_data.get(
        feature_a_id
    )
    right_data = cache_data.get(str(feature_b_id)) or cache_data.get(
        feature_b_id
    )

    if not left_data or not right_data:
        return True, dbc.Alert(
            "Feature data not found in cache.",
            color="warning",
        )

    # Get vocab mappings
    inv_vocab = cache_data.get("_inv_vocab", {})
    weapon_names = cache_data.get("_weapon_names", {})

    if not inv_vocab:
        inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    if not weapon_names and DASHBOARD_CONTEXT:
        analyzer = getattr(DASHBOARD_CONTEXT, "tfidf_analyzer", None)
        if analyzer:
            weapon_names = getattr(analyzer, "id_to_name_mapping", {})

    feature_a_name = left_data["display_name"]
    feature_b_name = right_data["display_name"]

    # Compute percentile ranges for both features
    # Top 10% (90-100%)
    top_range_a = compute_percentile_range_data(
        feature_a_id, feature_a_name, 90.0, 100.0, inv_vocab, weapon_names
    )
    top_range_b = compute_percentile_range_data(
        feature_b_id, feature_b_name, 90.0, 100.0, inv_vocab, weapon_names
    )

    # Middle 20% (40-60%)
    middle_range_a = compute_percentile_range_data(
        feature_a_id, feature_a_name, 40.0, 60.0, inv_vocab, weapon_names
    )
    middle_range_b = compute_percentile_range_data(
        feature_b_id, feature_b_name, 40.0, 60.0, inv_vocab, weapon_names
    )

    if (
        not top_range_a
        or not top_range_b
        or not middle_range_a
        or not middle_range_b
    ):
        return True, dbc.Alert(
            "Could not compute percentile ranges. Not enough activation data.",
            color="warning",
        )

    # Build comparison
    comparison = PercentileComparison(
        feature_a_id=feature_a_id,
        feature_b_id=feature_b_id,
        feature_a_name=feature_a_name,
        feature_b_name=feature_b_name,
        top_range_a=top_range_a,
        top_range_b=top_range_b,
        middle_range_a=middle_range_a,
        middle_range_b=middle_range_b,
    )

    # Build modal content
    content = build_percentile_comparison_content(comparison)

    return True, content
