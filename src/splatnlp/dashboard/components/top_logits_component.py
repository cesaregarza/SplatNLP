from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from dash import Input, Output, State, callback, dcc, html

# App context will be monkey-patched by the run script
# DASHBOARD_CONTEXT = None # This will be set by run_dashboard.py or cli.py

top_logits_component = html.Div(
    id="top-logits-content",
    children=[
        html.H4(
            "Top Output Logits Influenced by SAE Feature", className="mb-3"
        ),
        dcc.Loading(
            id="loading-top-logits",
            type="default",
            children=dcc.Graph(id="top-logits-graph"),
            className="mb-2",
        ),
        html.P(id="top-logits-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


@callback(
    [
        Output("top-logits-graph", "figure"),
        Output("top-logits-error-message", "children"),
    ],
    [Input("feature-dropdown", "value")],
)
def update_top_logits_graph(
    selected_feature_id: Optional[int],
) -> Tuple[Dict[str, Any], str]:
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return {
            "data": [],
            "layout": {"title": "Select a feature to see its logit influences"},
        }, ""

    if DASHBOARD_CONTEXT is None:
        return {
            "data": [],
            "layout": {"title": "Dashboard context not initialized"},
        }, "Error: Dashboard context not initialized"

    # Get logit influences from database
    influences = DASHBOARD_CONTEXT.db.get_logit_influences(selected_feature_id)

    if not influences["positive"] and not influences["negative"]:
        return {
            "data": [],
            "layout": {"title": "No logit influences found for this feature"},
        }, ""

    # Prepare data for plotting
    positive_tokens = [item["token_name"] for item in influences["positive"]]
    negative_tokens = [item["token_name"] for item in influences["negative"]]
    positive_effects = [
        item["influence_value"] for item in influences["positive"]
    ]
    negative_effects = [
        item["influence_value"] for item in influences["negative"]
    ]

    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            "Token": positive_tokens + negative_tokens,
            "Effect": positive_effects + negative_effects,
            "Type": ["Positive"] * len(positive_tokens)
            + ["Negative"] * len(negative_tokens),
        }
    )

    fig = px.bar(
        df,
        x="Token",
        y="Effect",
        color="Type",
        title=f"Top Logit Influences for Feature {selected_feature_id}",
        labels={"Token": "Token", "Effect": "Logit Influence"},
        color_discrete_map={"Positive": "#1f77b4", "Negative": "#ff7f0e"},
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
        xaxis={"tickangle": 45},  # Rotate labels for better readability
    )

    return fig, ""
