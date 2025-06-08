from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from dash import Input, Output, State, callback, dcc, html

from splatnlp.model.models import SetCompletionModel

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


def compute_logit_influences(
    feature_id: int,
    model: SetCompletionModel,
    inv_vocab: dict[str, str],
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Compute logit influences for a feature using the model's weights.

    Args:
        feature_id: The feature ID to compute influences for
        model: The SetCompletionModel containing the weights
        inv_vocab: Inverse vocabulary mapping token IDs to names
        limit: Number of top positive and negative influences to return

    Returns:
        Dictionary with 'positive' and 'negative' influence lists
    """
    # Get the output layer weights
    output_weights = (
        model.output_layer.weight.data.cpu().numpy()
    )  # Shape: (140, 512)

    # Get top positive and negative influences
    top_pos_indices = np.argsort(output_weights[:, feature_id])[-limit:][::-1]
    top_neg_indices = np.argsort(output_weights[:, feature_id])[:limit]

    # Convert to dictionaries
    positive = [
        {
            "token_id": int(idx),
            "token_name": inv_vocab.get(int(idx), f"Token_{idx}"),
            "influence_value": float(output_weights[idx, feature_id]),
        }
        for idx in top_pos_indices
    ]

    negative = [
        {
            "token_id": int(idx),
            "token_name": inv_vocab.get(int(idx), f"Token_{idx}"),
            "influence_value": float(output_weights[idx, feature_id]),
        }
        for idx in top_neg_indices
    ]

    return {"positive": positive, "negative": negative}


@callback(
    [
        Output("top-logits-graph", "figure"),
        Output("top-logits-error-message", "children"),
    ],
    [Input("feature-dropdown", "value")],
)
def update_top_logits_graph(
    selected_feature_id: int | None,
) -> tuple[dict[str, Any], str]:
    """Update the top logits graph when a feature is selected."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return {}, "No feature selected."

    if (
        not hasattr(DASHBOARD_CONTEXT, "primary_model")
        or DASHBOARD_CONTEXT.primary_model is None
    ):
        return (
            {},
            "Error: Model not available. Dynamic tooltips must be enabled.",
        )

    if not hasattr(DASHBOARD_CONTEXT, "inv_vocab"):
        return {}, "Error: Vocabulary not available."

    try:
        # Compute logit influences using model weights
        influences = compute_logit_influences(
            selected_feature_id,
            DASHBOARD_CONTEXT.primary_model,
            DASHBOARD_CONTEXT.inv_vocab,
        )

        # Prepare data for plotting
        positive_df = pd.DataFrame(influences["positive"])
        negative_df = pd.DataFrame(influences["negative"])

        # Create figure
        fig = px.bar(
            pd.concat(
                [
                    positive_df.assign(direction="Positive"),
                    negative_df.assign(direction="Negative"),
                ]
            ),
            x="token_name",
            y="influence_value",
            color="direction",
            barmode="group",
            title=f"Top Logit Influences for Feature {selected_feature_id}",
            labels={
                "token_name": "Token",
                "influence_value": "Influence Value",
                "direction": "Direction",
            },
        )

        return fig, ""

    except Exception as e:
        return {}, f"Error computing logit influences: {str(e)}"
