from typing import Any, Dict, Optional, Tuple

import numpy as np
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

    sae_model = DASHBOARD_CONTEXT.sae_model
    if sae_model is None:
        return {
            "data": [],
            "layout": {"title": "SAE model not loaded"},
        }, "Error: SAE model not loaded"

    if not hasattr(sae_model, "decoder") or not hasattr(
        sae_model.decoder, "weight"
    ):
        return {
            "data": [],
            "layout": {"title": "SAE model missing decoder weights"},
        }, "Error: SAE model missing decoder weights"

    if not (0 <= selected_feature_id < sae_model.decoder.weight.shape[1]):
        return {
            "data": [],
            "layout": {
                "title": f"Feature ID {selected_feature_id} out of range"
            },
        }, f"Error: Feature ID {selected_feature_id} out of range"

    sae_feature_direction = (
        sae_model.decoder.weight.data.detach()
        .cpu()
        .numpy()[:, selected_feature_id]
    )

    primary_model = DASHBOARD_CONTEXT.primary_model
    if primary_model is None:
        return {
            "data": [],
            "layout": {"title": "Primary model not loaded"},
        }, "Error: Primary model not loaded"

    if not hasattr(primary_model, "output_layer") or not hasattr(
        primary_model.output_layer, "weight"
    ):
        return {
            "data": [],
            "layout": {"title": "Primary model missing output layer weights"},
        }, "Error: Primary model missing output layer weights"

    primary_output_layer_weights = (
        primary_model.output_layer.weight.data.detach().cpu().numpy()
    )
    sae_feature_effect_on_primary_activation = (
        sae_feature_direction  # This is already a numpy array
    )

    # Ensure DEVICE is available in DASHBOARD_CONTEXT, otherwise default to 'cpu'
    DEVICE = getattr(DASHBOARD_CONTEXT, "device", "cpu")
    if DEVICE is None:  # Handle case where device might be explicitly None
        DEVICE = "cpu"
        print(
            "Warning: DASHBOARD_CONTEXT.device was None, defaulting to 'cpu' for top_logits_component."
        )

    # Convert numpy arrays to tensors and move to the correct device
    t_primary_output_layer_weights = torch.as_tensor(
        primary_output_layer_weights, dtype=torch.float32
    ).to(DEVICE)
    t_sae_feature_effect_on_primary_activation = torch.as_tensor(
        sae_feature_effect_on_primary_activation, dtype=torch.float32
    ).to(DEVICE)

    # Perform matrix multiplication
    # primary_output_layer_weights is likely (output_vocab_size, hidden_dim)
    # sae_feature_effect_on_primary_activation is likely (hidden_dim,)
    # Resulting effect_on_logits should be (output_vocab_size,)
    effect_on_logits = torch.matmul(
        t_primary_output_layer_weights,
        t_sae_feature_effect_on_primary_activation,
    )

    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    if inv_vocab is None:
        return {
            "data": [],
            "layout": {"title": "Vocabulary not loaded"},
        }, "Error: Vocabulary not loaded"

    token_names = [
        inv_vocab.get(str(i), inv_vocab.get(i, f"Token_ID_{i}"))
        for i in range(len(effect_on_logits))
    ]

    sorted_indices = torch.argsort(effect_on_logits)
    top_positive = sorted_indices[-5:].flip(0)  # Changed from -10 to -5
    top_negative = sorted_indices[:5]  # Changed from :10 to :5

    positive_tokens = [token_names[i] for i in top_positive]
    negative_tokens = [token_names[i] for i in top_negative]
    positive_effects = [effect_on_logits[i].item() for i in top_positive]
    negative_effects = [effect_on_logits[i].item() for i in top_negative]

    fig = px.bar(
        x=positive_tokens + negative_tokens,
        y=positive_effects + negative_effects,
        title=f"Top Logit Influences for Feature {selected_feature_id}",
        labels={"x": "Token", "y": "Logit Influence"},
        color=["Positive"] * 5 + ["Negative"] * 5,  # Changed from 10 to 5
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,  # Set fixed height for the graph
    )

    return fig, ""
