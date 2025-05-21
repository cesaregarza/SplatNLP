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
def update_top_logits_graph(selected_feature_id):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return {
            "data": [],
            "layout": {
                "title": "Select an SAE feature to see logit influences"
            },
        }, ""
    if DASHBOARD_CONTEXT is None:
        return {
            "data": [],
            "layout": {"title": "Dashboard context not loaded"},
        }, "Error: DASHBOARD_CONTEXT is None. Ensure data is loaded."

    sae_model = DASHBOARD_CONTEXT.sae_model
    primary_model = DASHBOARD_CONTEXT.primary_model
    inv_vocab = (
        DASHBOARD_CONTEXT.inv_vocab
    )  # Expected: {int_index: token_name_str}

    if not hasattr(sae_model, "decoder") or not hasattr(
        sae_model.decoder, "weight"
    ):
        return {
            "data": [],
            "layout": {"title": "SAE decoder weights not found"},
        }, "Error: SAE decoder or its weights not found."
    if not hasattr(sae_model, "hidden_dim"):
        return {
            "data": [],
            "layout": {"title": "SAE hidden_dim not found"},
        }, "Error: SAE hidden_dim attribute not found on sae_model."

    if not hasattr(primary_model, "output_layer") or not hasattr(
        primary_model.output_layer, "weight"
    ):
        return {
            "data": [],
            "layout": {"title": "Primary model output layer weights not found"},
        }, "Error: Primary model output_layer or its weights not found."

    if selected_feature_id >= sae_model.hidden_dim:
        err_msg = f"Error: Feature ID {selected_feature_id} is out of range for SAE hidden dim {sae_model.hidden_dim}."
        return {"data": [], "layout": {"title": err_msg}}, err_msg

    # Get SAE decoder weights for the selected feature
    # sae_model.decoder: nn.Linear(self.hidden_dim, input_dim) -> decoder.weight: (input_dim, hidden_dim)
    # We need the column vector corresponding to the selected_feature_id.
    try:
        sae_feature_effect_on_primary_activation = (
            sae_model.decoder.weight.data[:, selected_feature_id].detach().cpu()
        )
    except IndexError:
        err_msg = f"Error: Feature ID {selected_feature_id} caused an IndexError when accessing SAE decoder weights. Max index is {sae_model.decoder.weight.data.shape[1]-1}."
        return {
            "data": [],
            "layout": {"title": "SAE Weight Access Error"},
        }, err_msg

    # primary_model.output_layer: nn.Linear(hidden_dim_primary, output_dim_vocab)
    # primary_model.output_layer.weight: (output_dim_vocab, hidden_dim_primary)
    primary_output_layer_weights = (
        primary_model.output_layer.weight.data.detach().cpu()
    )

    if (
        sae_feature_effect_on_primary_activation.shape[0]
        != primary_output_layer_weights.shape[1]
    ):
        err_msg = (
            f"Dimension mismatch: SAE feature effect vector size ({sae_feature_effect_on_primary_activation.shape[0]}) "
            f"does not match primary model output layer input dimension ({primary_output_layer_weights.shape[1]})."
        )
        return {
            "data": [],
            "layout": {"title": "Dimension Mismatch"},
        }, "Error: " + err_msg

    # (vocab_size, hidden_dim_primary) @ (hidden_dim_primary) -> (vocab_size)
    effect_on_logits = torch.matmul(
        primary_output_layer_weights, sae_feature_effect_on_primary_activation
    )
    effect_on_logits_np = effect_on_logits.numpy()

    top_n = 15
    if effect_on_logits_np.ndim > 1:
        effect_on_logits_np = effect_on_logits_np.squeeze()
    if effect_on_logits_np.ndim == 0:
        effect_on_logits_np = np.array([effect_on_logits_np.item()])

    sorted_indices = np.argsort(effect_on_logits_np)

    num_logits = len(effect_on_logits_np)
    actual_top_n = min(top_n, num_logits)

    top_positive_indices = sorted_indices[-actual_top_n:][::-1]
    top_negative_indices = sorted_indices[:actual_top_n]

    # Handle cases where there might be overlap or fewer than 2*top_n unique tokens
    # For example, if vocab size is small or effects are very concentrated
    combined_indices_set = set(top_positive_indices) | set(top_negative_indices)
    combined_indices = sorted(
        list(combined_indices_set),
        key=lambda x: effect_on_logits_np[x],
        reverse=True,
    )

    results = []
    for idx in combined_indices:
        token_idx_int = int(idx)
        token_name = inv_vocab.get(token_idx_int, f"Token_ID_{token_idx_int}")

        # If inv_vocab keys might be strings (e.g. from a JSON load where ints became strings)
        if (
            token_name == f"Token_ID_{token_idx_int}"
        ):  # Fallback if int key didn't work
            token_name = inv_vocab.get(
                str(token_idx_int), f"Token_ID_{token_idx_int}"
            )

        results.append(
            {
                "token": token_name,
                "effect": effect_on_logits_np[idx],
                "type": (
                    "Positive" if effect_on_logits_np[idx] >= 0 else "Negative"
                ),
            }
        )

    if not results:
        return {
            "data": [],
            "layout": {
                "title": f"No significant logit effects found for Feature {selected_feature_id}"
            },
        }, ""

    fig = px.bar(
        results,
        x="token",
        y="effect",
        color="type",
        title=f"Top Logit Influences for SAE Feature {selected_feature_id}",
        labels={"effect": "Influence on Logit Value", "token": "Output Token"},
        color_discrete_map={"Positive": "green", "Negative": "red"},
    )
    fig.update_layout(xaxis_categoryorder="total descending")

    return fig, ""
