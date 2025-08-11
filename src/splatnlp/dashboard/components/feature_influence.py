"""Feature influence component for dashboard."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.dash_table import DataTable

from splatnlp.dashboard.fs_database import FSDatabase

feature_influence_component = html.Div(
    [
        html.H4("Feature Influence on Output Tokens", className="mb-3"),
        html.Div(
            [
                html.P(
                    "Shows how this feature influences the model's output logits - "
                    "which tokens become more or less likely when this feature is active.",
                    className="text-muted small",
                ),
                dcc.Loading(
                    id="loading-influence",
                    type="default",
                    children=[
                        html.Div(id="influence-summary"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            "Positive Influence (More Likely)",
                                            className="text-success",
                                        ),
                                        html.Div(
                                            id="positive-influence-table",
                                            style={
                                                "maxHeight": "500px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                    ],
                                    className="col-md-6",
                                ),
                                html.Div(
                                    [
                                        html.H5(
                                            "Negative Influence (Less Likely)",
                                            className="text-danger",
                                        ),
                                        html.Div(
                                            id="negative-influence-table",
                                            style={
                                                "maxHeight": "500px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                    ],
                                    className="col-md-6",
                                ),
                            ],
                            className="row mt-3",
                        ),
                        dcc.Graph(
                            id="influence-chart", style={"height": "400px"}
                        ),
                    ],
                ),
            ],
        ),
    ],
    className="mb-4",
)


@callback(
    [
        Output("influence-summary", "children"),
        Output("positive-influence-table", "children"),
        Output("negative-influence-table", "children"),
        Output("influence-chart", "figure"),
    ],
    [Input("feature-dropdown", "value")],
    State("feature-dropdown", "value"),
)
def update_feature_influence(
    selected_feature_id: int | None, _: int | None
) -> tuple[Any, Any, Any, dict]:
    """Update feature influence displays."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    empty_return = (
        html.Div("Select a feature to view influence data"),
        html.Div(),
        html.Div(),
        {"data": [], "layout": {}},
    )

    if selected_feature_id is None:
        return empty_return

    if (
        DASHBOARD_CONTEXT is None
        or not hasattr(DASHBOARD_CONTEXT, "influence_data")
        or DASHBOARD_CONTEXT.influence_data is None
    ):
        return (
            html.Div(
                "Influence data not available. Please run precomputation script.",
                className="alert alert-warning",
            ),
            html.Div(),
            html.Div(),
            {"data": [], "layout": {}},
        )

    # Get precomputed influence data
    influence_df = DASHBOARD_CONTEXT.influence_data

    # Filter for selected feature
    feature_data = influence_df[
        influence_df["feature_id"] == selected_feature_id
    ]

    if feature_data.empty:
        return (
            html.Div(
                f"No influence data for Feature {selected_feature_id}",
                className="alert alert-info",
            ),
            html.Div(),
            html.Div(),
            {"data": [], "layout": {}},
        )

    feature_row = feature_data.iloc[0]

    # Extract positive and negative influences
    pos_data = []
    neg_data = []

    # Dynamically determine max available influences
    max_influences = 100  # Try to load up to 100

    for i in range(1, max_influences + 1):
        pos_tok_col = f"+{i}_tok"
        pos_val_col = f"+{i}_val"
        neg_tok_col = f"-{i}_tok"
        neg_val_col = f"-{i}_val"

        # Stop if we've reached the end of available data
        if pos_tok_col not in feature_row and neg_tok_col not in feature_row:
            if i <= 30:  # If we don't even have 30, there might be an issue
                continue
            else:
                break  # We've reached the end of available influences

        if pos_tok_col in feature_row and pd.notna(feature_row[pos_tok_col]):
            pos_data.append(
                {
                    "Rank": i,
                    "Token": feature_row[pos_tok_col],
                    "Influence": f"{feature_row[pos_val_col]:.4f}",
                }
            )

        if neg_tok_col in feature_row and pd.notna(feature_row[neg_tok_col]):
            neg_data.append(
                {
                    "Rank": i,
                    "Token": feature_row[neg_tok_col],
                    "Influence": f"{feature_row[neg_val_col]:.4f}",
                }
            )

    # Calculate summary statistics based on actual available data
    pos_sum = sum(
        float(feature_row[f"+{i}_val"])
        for i in range(1, max_influences + 1)
        if f"+{i}_val" in feature_row and pd.notna(feature_row[f"+{i}_val"])
    )
    neg_sum = sum(
        float(feature_row[f"-{i}_val"])
        for i in range(1, max_influences + 1)
        if f"-{i}_val" in feature_row and pd.notna(feature_row[f"-{i}_val"])
    )
    net_influence = pos_sum + neg_sum

    summary = html.Div(
        [
            html.P(
                [
                    html.Strong("Feature: "),
                    f"{selected_feature_id} - {feature_row.get('feature_label', 'Unlabeled')}",
                ],
                className="mb-2",
            ),
            html.P(
                [
                    html.Span(
                        f"Positive Sum: {pos_sum:.4f}",
                        className="badge bg-success me-2",
                    ),
                    html.Span(
                        f"Negative Sum: {neg_sum:.4f}",
                        className="badge bg-danger me-2",
                    ),
                    html.Span(
                        f"Net: {net_influence:.4f}",
                        className=(
                            "badge bg-primary"
                            if abs(net_influence) < 0.001
                            else "badge bg-warning"
                        ),
                    ),
                ],
                className="mb-2",
            ),
            html.P(
                [
                    html.Small(
                        f"Showing {len(pos_data)} positive and {len(neg_data)} negative influences",
                        className="text-muted",
                    ),
                ],
                className="mb-0",
            ),
        ]
    )

    # Create tables - show all available data (up to 30 items)
    pos_table = DataTable(
        data=pos_data,  # Show all available positive influences
        columns=[
            {"name": "Rank", "id": "Rank"},
            {"name": "Token", "id": "Token"},
            {"name": "Influence", "id": "Influence"},
        ],
        style_cell={"textAlign": "left"},
        style_data_conditional=[
            {
                "if": {"row_index": 0},
                "backgroundColor": "rgba(0, 255, 0, 0.1)",
            },
            {
                "if": {"row_index": 1},
                "backgroundColor": "rgba(0, 255, 0, 0.08)",
            },
            {
                "if": {"row_index": 2},
                "backgroundColor": "rgba(0, 255, 0, 0.06)",
            },
        ],
        style_table={"height": "auto", "overflowY": "auto"},
    )

    neg_table = DataTable(
        data=neg_data,  # Show all available negative influences
        columns=[
            {"name": "Rank", "id": "Rank"},
            {"name": "Token", "id": "Token"},
            {"name": "Influence", "id": "Influence"},
        ],
        style_cell={"textAlign": "left"},
        style_data_conditional=[
            {
                "if": {"row_index": 0},
                "backgroundColor": "rgba(255, 0, 0, 0.1)",
            },
            {
                "if": {"row_index": 1},
                "backgroundColor": "rgba(255, 0, 0, 0.08)",
            },
            {
                "if": {"row_index": 2},
                "backgroundColor": "rgba(255, 0, 0, 0.06)",
            },
        ],
        style_table={"height": "auto", "overflowY": "auto"},
    )

    # Create visualization
    fig = go.Figure()

    # Add positive influences
    if pos_data:
        pos_tokens = [d["Token"] for d in pos_data[:15]]
        pos_values = [float(d["Influence"]) for d in pos_data[:15]]
        fig.add_trace(
            go.Bar(
                x=pos_values,
                y=pos_tokens,
                orientation="h",
                name="Positive Influence",
                marker_color="green",
                text=[f"{v:.4f}" for v in pos_values],
                textposition="outside",
            )
        )

    # Add negative influences
    if neg_data:
        neg_tokens = [d["Token"] for d in neg_data[:15]]
        neg_values = [float(d["Influence"]) for d in neg_data[:15]]
        fig.add_trace(
            go.Bar(
                x=neg_values,
                y=neg_tokens,
                orientation="h",
                name="Negative Influence",
                marker_color="red",
                text=[f"{v:.4f}" for v in neg_values],
                textposition="outside",
            )
        )

    fig.update_layout(
        title=f"Top Token Influences for Feature {selected_feature_id}",
        xaxis_title="Influence Value",
        yaxis_title="Token",
        barmode="overlay",
        height=400,
        margin=dict(l=100, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return summary, pos_table, neg_table, fig
