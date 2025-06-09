import dash
import dash_bootstrap_components as dbc
import torch
from dash import Input, Output, State, callback, dcc, html


def _compute_feature_activation(
    build_tokens: list[str], weapon_name: str, feature_id: int
):
    """Compute SAE feature activation for a given build and weapon."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        feature_id is None
        or not hasattr(DASHBOARD_CONTEXT, "primary_model")
        or DASHBOARD_CONTEXT.primary_model is None
        or not hasattr(DASHBOARD_CONTEXT, "sae_model")
        or DASHBOARD_CONTEXT.sae_model is None
    ):
        return None

    vocab = DASHBOARD_CONTEXT.vocab
    weapon_vocab = DASHBOARD_CONTEXT.weapon_vocab
    device = getattr(DASHBOARD_CONTEXT, "device", "cpu")
    pad_id = vocab.get("<PAD>", 0)

    token_ids = [vocab.get(tok, pad_id) for tok in build_tokens]
    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)
    weapon_token = torch.tensor(
        [weapon_vocab.get(weapon_name, 0)], device=device
    ).unsqueeze(0)
    mask = tokens == pad_id

    model = DASHBOARD_CONTEXT.primary_model
    sae = DASHBOARD_CONTEXT.sae_model

    with torch.no_grad():
        ability_embeddings = model.ability_embedding(tokens)
        weapon_embeddings = model.weapon_embedding(weapon_token).expand_as(
            ability_embeddings
        )
        embeddings = ability_embeddings + weapon_embeddings
        x = model.input_proj(embeddings)
        for layer in model.transformer_layers:
            x = layer(x, key_padding_mask=mask)
        masked = model.masked_mean(x, mask)
        _, hidden = sae.encode(masked)
    return hidden[0, feature_id].item()


ablation_component = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(id="ablation-weapon", options=[]),
                    width=3,
                ),
                dbc.Col(
                    dcc.Input(
                        id="ablation-build",
                        type="text",
                        placeholder="Ability tokens space separated",
                        style={"width": "100%"},
                    ),
                    width=5,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            "Set as Primary",
                            id="ablation-primary-btn",
                            color="primary",
                            className="me-2",
                        ),
                        dbc.Button(
                            "Set as Secondary",
                            id="ablation-secondary-btn",
                            color="secondary",
                        ),
                    ],
                    width="auto",
                ),
            ],
            className="mb-3",
        ),
        dcc.Store(id="ablation-primary-store"),
        dcc.Store(id="ablation-secondary-store"),
        dcc.Store(id="ablation-load-store"),
        html.Div(id="ablation-results"),
    ]
)


@callback(
    Output("ablation-weapon", "options"), Input("page-load-trigger", "data")
)
def _populate_weapon_dropdown(_timestamp):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "weapon_vocab")
        or DASHBOARD_CONTEXT.weapon_vocab is None
    ):
        return []
    return [
        {"label": name, "value": name}
        for name in sorted(DASHBOARD_CONTEXT.weapon_vocab.keys())
    ]


@callback(
    Output("ablation-primary-store", "data"),
    Output("ablation-results", "children"),
    Input("ablation-primary-btn", "n_clicks"),
    State("ablation-build", "value"),
    State("ablation-weapon", "value"),
    State("feature-dropdown", "value"),
    prevent_initial_call=True,
)
def _set_primary(_, build_text, weapon_name, feature_id):
    build_tokens = build_text.split() if build_text else []
    act = _compute_feature_activation(build_tokens, weapon_name, feature_id)
    if act is None:
        return dash.no_update, "Model context unavailable"
    return act, f"Primary activation set: {act:.4f}"


@callback(
    Output("ablation-secondary-store", "data"),
    Output("ablation-results", "children", allow_duplicate=True),
    Input("ablation-secondary-btn", "n_clicks"),
    State("ablation-build", "value"),
    State("ablation-weapon", "value"),
    State("feature-dropdown", "value"),
    prevent_initial_call=True,
)
def _set_secondary(_, build_text, weapon_name, feature_id):
    build_tokens = build_text.split() if build_text else []
    act = _compute_feature_activation(build_tokens, weapon_name, feature_id)
    if act is None:
        return dash.no_update, "Model context unavailable"
    return act, f"Secondary activation set: {act:.4f}"


@callback(
    Output("ablation-results", "children", allow_duplicate=True),
    Input("ablation-primary-store", "data"),
    Input("ablation-secondary-store", "data"),
    prevent_initial_call=True,
)
def _display_difference(primary_act, secondary_act):
    if primary_act is None or secondary_act is None:
        return dash.no_update
    diff = secondary_act - primary_act
    return f"Difference: {diff:.4f}"


@callback(
    Output("ablation-build", "value"),
    Output("ablation-weapon", "value"),
    Input("ablation-load-store", "data"),
    prevent_initial_call=True,
)
def _load_from_store(data):
    if not data:
        raise dash.exceptions.PreventUpdate
    return " ".join(data.get("build_tokens", [])), data.get("weapon_name")
