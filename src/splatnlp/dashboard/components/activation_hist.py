from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from splatnlp.dashboard.fs_database import FSDatabase

# App context will be monkey-patched by the run script
# We need to declare it here for the callback to access it.
# This is a common pattern in multi-file Dash apps.
# See: https://dash.plotly.com/sharing-data-between-callbacks
# DASHBOARD_CONTEXT = None # This will be set by run_dashboard.py or cli.py

activation_hist_component = html.Div(
    [
        html.H4("Feature Activation Histogram", className="mb-3"),
        dcc.RadioItems(
            id="activation-filter-radio",
            options=[
                {"label": "All Activations", "value": "all"},
                {"label": "Non-Zero Activations", "value": "nonzero"},
            ],
            value="nonzero",
            labelStyle={"display": "inline-block", "margin-right": "10px"},
            className="mb-2",
        ),
        dcc.Graph(id="activation-histogram"),
    ],
    className="mb-4",
)


@callback(
    Output("activation-histogram", "figure"),
    [
        Input("feature-dropdown", "value"),
        Input("activation-filter-radio", "value"),
    ],
    State(
        "feature-dropdown", "value"
    ),  # Keep state for consistency, though not strictly needed if using selected_feature_id
)
def update_activation_histogram(
    selected_feature_id: int | None,
    filter_type: str,
    _: int | None,  # State variable, not used directly
) -> dict[str, Any]:  # Return type is now dict for consistency with old style
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return {"data": [], "layout": {}}

    if (
        DASHBOARD_CONTEXT is None
        or not hasattr(DASHBOARD_CONTEXT, "db")
        or DASHBOARD_CONTEXT.db is None
    ):
        return {
            "data": [],
            "layout": {"title": "Error: Database context not available."},
        }

    db: FSDatabase = DASHBOARD_CONTEXT.db
    histogram_df = db.get_feature_histogram(selected_feature_id)

    if histogram_df.is_empty():
        return {
            "data": [],
            "layout": {
                "title": f"No histogram data for Feature {selected_feature_id}"
            },
        }

    # Filter bins if needed
    if filter_type == "nonzero":
        histogram_df = histogram_df[histogram_df["lower_bound"] > 1e-6]

    if histogram_df.empty:
        return {
            "data": [],
            "layout": {
                "title": f"No strictly positive activation bins for Feature {selected_feature_id}"
            },
        }

    # Create the histogram directly from the binned data
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=(histogram_df["lower_bound"] + histogram_df["upper_bound"]) / 2,
            y=histogram_df["count"],
            width=histogram_df["upper_bound"] - histogram_df["lower_bound"],
            hovertext=[
                f"Range: [{min_act:.3g} - {max_act:.3g})<br>Count: {count}"
                for min_act, max_act, count in zip(
                    histogram_df["lower_bound"],
                    histogram_df["upper_bound"],
                    histogram_df["count"],
                )
            ],
            hoverinfo="text",
            name="Count",
        )
    )

    fig.update_layout(
        title=f"Activation Distribution for Feature {selected_feature_id} {('(Non-Zero Activations)' if filter_type == 'nonzero' else '')}",
        xaxis_title="Activation Value",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        bargap=0.1,
    )

    return fig
