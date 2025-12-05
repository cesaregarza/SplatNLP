import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html, no_update

from splatnlp.model.models import SetCompletionModel

HIGH_AP_PATTERN = re.compile(r"_(21|29|38|51|57)$")
SPECIAL_TOKENS = {"<PAD>", "<NULL>"}

top_logits_component = html.Div(
    id="top-logits-content",
    children=[
        html.H4(
            "Top Output Logits Influenced by SAE Feature", className="mb-3"
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Positive Influences", className="mb-2"),
                        dcc.Loading(
                            id="loading-top-positive-logits",
                            type="default",
                            children=dcc.Graph(id="top-positive-logits-graph"),
                            className="mb-2",
                        ),
                    ],
                    className="col-6",
                ),
                html.Div(
                    [
                        html.H5("Negative Influences", className="mb-2"),
                        dcc.RadioItems(
                            id="filter-tokens-radio",
                            options=[
                                {"label": "Show All Tokens", "value": "all"},
                                {
                                    "label": "Filter High AP Tokens",
                                    "value": "filter",
                                },
                            ],
                            value="all",
                            labelStyle={
                                "display": "inline-block",
                                "margin-right": "10px",
                            },
                            className="mb-2",
                        ),
                        dcc.Loading(
                            id="loading-top-negative-logits",
                            type="default",
                            children=dcc.Graph(id="top-negative-logits-graph"),
                            className="mb-2",
                        ),
                    ],
                    className="col-6",
                ),
            ],
            className="row",
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
    output_weights = model.output_layer.weight.data.cpu().numpy()

    top_pos_indices = np.argsort(output_weights[:, feature_id])[-limit * 3 :][
        ::-1
    ]
    top_neg_indices = np.argsort(output_weights[:, feature_id])[: limit * 10]

    positive = [
        {
            "token_id": int(idx),
            "token_name": inv_vocab.get(int(idx), f"Token_{idx}"),
            "influence_value": float(output_weights[idx, feature_id]),
        }
        for idx in top_pos_indices
        if inv_vocab.get(int(idx), f"Token_{idx}") not in SPECIAL_TOKENS
    ]

    negative = [
        {
            "token_id": int(idx),
            "token_name": inv_vocab.get(int(idx), f"Token_{idx}"),
            "influence_value": float(output_weights[idx, feature_id]),
        }
        for idx in top_neg_indices
        if inv_vocab.get(int(idx), f"Token_{idx}") not in SPECIAL_TOKENS
    ]

    return {"positive": positive, "negative": negative}


@callback(
    [
        Output("top-positive-logits-graph", "figure"),
        Output("top-negative-logits-graph", "figure"),
        Output("top-logits-error-message", "children"),
    ],
    [
        Input("feature-dropdown", "value"),
        Input("filter-tokens-radio", "value"),
        Input("active-tab-store", "data"),
    ],
)
def update_top_logits_graph(
    selected_feature_id: int | None,
    filter_type: str,
    active_tab: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Update the top logits graphs when a feature is selected."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Lazy loading: skip if tab is not active
    if active_tab != "tab-logits":
        return no_update, no_update, no_update

    if selected_feature_id is None:
        return {}, {}, "No feature selected."

    if (
        not hasattr(DASHBOARD_CONTEXT, "primary_model")
        or DASHBOARD_CONTEXT.primary_model is None
    ):
        return (
            {},
            {},
            "Error: Model not available. Dynamic tooltips must be enabled.",
        )

    if not hasattr(DASHBOARD_CONTEXT, "inv_vocab"):
        return {}, {}, "Error: Vocabulary not available."

    try:
        influences = compute_logit_influences(
            selected_feature_id,
            DASHBOARD_CONTEXT.primary_model,
            DASHBOARD_CONTEXT.inv_vocab,
            limit=10,
        )

        positive_df = pd.DataFrame(influences["positive"])
        negative_df = pd.DataFrame(influences["negative"])

        if filter_type == "filter":
            positive_df = positive_df[
                ~positive_df["token_name"].str.contains(
                    HIGH_AP_PATTERN, regex=True
                )
            ]
            negative_df = negative_df[
                ~negative_df["token_name"].str.contains(
                    HIGH_AP_PATTERN, regex=True
                )
            ]

        positive_df = positive_df.head(10)
        negative_df = negative_df.head(10)

        positive_fig = px.bar(
            positive_df,
            x="token_name",
            y="influence_value",
            title=f"Top Positive Logit Influences for Feature {selected_feature_id}",
            labels={
                "token_name": "Token",
                "influence_value": "Influence Value",
            },
            color_discrete_sequence=["#2ecc71"],
        )

        negative_fig = px.bar(
            negative_df,
            x="token_name",
            y="influence_value",
            title=f"Top Negative Logit Influences for Feature {selected_feature_id}",
            labels={
                "token_name": "Token",
                "influence_value": "Influence Value",
            },
            color_discrete_sequence=["#e74c3c"],
        )

        for fig in [positive_fig, negative_fig]:
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
            )

        return positive_fig, negative_fig, ""

    except Exception as e:
        return {}, {}, f"Error computing logit influences: {str(e)}"
