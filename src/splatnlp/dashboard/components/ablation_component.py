import json  # For potentially displaying dicts or if data is stored as JSON string

import dash
import dash_bootstrap_components as dbc
import torch  # Import PyTorch
from dash import Input, Output, State, callback_context, dcc, html

# Import weapon name mapping utility
from splatnlp.dashboard.utils.converters import generate_weapon_name_mapping

# Define the layout for the Ablation tab
layout = html.Div(
    [
        dcc.Store(id="ablation-primary-store"),  # Store for primary build data
        dcc.Store(
            id="ablation-secondary-store"
        ),  # Store for modified/secondary build data
        html.H3("Ablation Analysis"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Primary Build Details"),
                        html.Div(
                            id="primary-build-display",
                            children="No primary build selected.",
                            className="mb-3 p-2 border rounded bg-light",
                            style={"minHeight": "100px"},
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.H4("Secondary Build Input (Modify Build)"),
                        html.Label("Select Weapon:"),
                        dcc.Dropdown(
                            id="secondary-weapon-dropdown",
                            options=[],
                            placeholder="Select weapon (or keep original)...",
                            className="mb-2",
                        ),
                        html.Label("Select Abilities:"),
                        dcc.Dropdown(
                            id="secondary-build-input",
                            options=[],
                            multi=True,
                            placeholder="Select abilities...",
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Run Ablation Analysis",
                            id="run-ablation-button",
                            color="success",
                            className="mt-2",
                        ),
                    ],
                    md=6,
                ),
            ]
        ),
        html.Hr(),
        html.H4("Ablation Results"),
        html.Div(
            id="ablation-results-display",
            children="Ablation results will appear here.",  # Placeholder
            className="mt-3 p-2 border rounded",
        ),
    ]
)


# Callback to populate ability dropdown options
@dash.callback(
    Output("secondary-build-input", "options"),
    Input("page-load-trigger", "data"),
)
def populate_ability_dropdown(_):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "vocab")
        or DASHBOARD_CONTEXT.vocab is None
    ):
        return []

    # Get all ability tokens (exclude special tokens that start with <)
    return [
        {"label": tok, "value": tok}
        for tok in sorted(DASHBOARD_CONTEXT.vocab.keys())
        if not tok.startswith("<")
    ]


# Callback to populate weapon dropdown with English names
@dash.callback(
    Output("secondary-weapon-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_weapon_dropdown(_):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab")
        or DASHBOARD_CONTEXT.inv_weapon_vocab is None
    ):
        return []

    # Generate weapon name mapping to get English names
    weapon_name_mapping = generate_weapon_name_mapping(
        DASHBOARD_CONTEXT.inv_weapon_vocab
    )

    # Create options with English names as labels and weapon IDs as values
    options = []
    for weapon_id, raw_name in DASHBOARD_CONTEXT.inv_weapon_vocab.items():
        english_name = weapon_name_mapping.get(weapon_id, raw_name)
        options.append(
            {
                "label": english_name,
                "value": weapon_id,  # Store the ID as the value
            }
        )

    # Sort by English name
    options.sort(key=lambda x: x["label"])

    return options


# Callback to display primary build details and pre-fill secondary input
@dash.callback(
    [
        Output("primary-build-display", "children"),
        Output("secondary-build-input", "value"),
        Output("secondary-weapon-dropdown", "value"),
    ],
    Input("ablation-primary-store", "data"),
)
def display_primary_build(primary_data):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT  # Moved import here

    if primary_data:
        inv_vocab = (
            DASHBOARD_CONTEXT.inv_vocab
            if hasattr(DASHBOARD_CONTEXT, "inv_vocab")
            else {}
        )
        inv_weapon_vocab = (
            DASHBOARD_CONTEXT.inv_weapon_vocab
            if hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab")
            else {}
        )
        weapon_id = primary_data.get("weapon_id_token", "N/A")
        ability_token_ids = primary_data.get("ability_input_tokens", [])
        activation = primary_data.get("activation", "N/A")

        ability_names = [
            str(inv_vocab.get(token_id, token_id))
            for token_id in ability_token_ids
        ]
        abilities_display_str = (
            ", ".join(ability_names) if ability_names else "None"
        )

        # Get English weapon name from ID
        weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
        weapon_raw_name = inv_weapon_vocab.get(weapon_id, f"Weapon {weapon_id}")
        weapon_english_name = weapon_name_mapping.get(
            weapon_id, weapon_raw_name
        )

        display_children = [
            html.P(f"Weapon: {weapon_english_name}"),
            html.P(f"Abilities: {abilities_display_str}"),
            html.P(
                f"Original Activation: {activation:.4f}"
                if isinstance(activation, float)
                else f"Original Activation: {activation}"
            ),
        ]
        return (
            display_children,
            ability_names,  # Pre-fill abilities
            weapon_id,  # Pre-fill weapon with same weapon
        )
    else:
        return "No primary build selected.", [], None


# Callback to update the secondary store from the dropdown
@dash.callback(
    Output("ablation-secondary-store", "data"),
    Input("secondary-build-input", "value"),
)
def update_secondary_store(secondary_build_list):
    # Store the list of ability names
    if secondary_build_list is None:
        return dash.no_update
    return {"ability_tokens_list": secondary_build_list}


# Helper function to compute feature activation
@torch.no_grad()
def compute_feature_activation(
    ability_names: list[str], weapon_name: str, feature_id: int
):
    """Compute SAE feature activation for a given build and weapon."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Debug logging (comment out for production)
    # print(f"DEBUG: compute_feature_activation called with:")
    # print(f"  ability_names: {ability_names}")
    # print(f"  weapon_name: {weapon_name}")
    # print(f"  feature_id: {feature_id}")

    if feature_id is None:
        print("ERROR: feature_id is None")
        return None

    if not hasattr(DASHBOARD_CONTEXT, "primary_model"):
        print("ERROR: DASHBOARD_CONTEXT has no primary_model")
        return None

    if DASHBOARD_CONTEXT.primary_model is None:
        print("ERROR: primary_model is None")
        return None

    if not hasattr(DASHBOARD_CONTEXT, "sae_model"):
        print("ERROR: DASHBOARD_CONTEXT has no sae_model")
        return None

    if DASHBOARD_CONTEXT.sae_model is None:
        print("ERROR: sae_model is None")
        return None

    vocab = DASHBOARD_CONTEXT.vocab
    weapon_vocab = DASHBOARD_CONTEXT.weapon_vocab
    device = getattr(DASHBOARD_CONTEXT, "device", "cpu")
    pad_id = vocab.get("<PAD>", 0)

    # Convert ability names to token IDs
    token_ids = [vocab.get(tok, pad_id) for tok in ability_names]

    # Check weapon lookup
    if weapon_name not in weapon_vocab:
        print(f"ERROR: Weapon '{weapon_name}' not in weapon_vocab")
        print(f"  Available weapons (first 5): {list(weapon_vocab.keys())[:5]}")
        return None

    weapon_id = weapon_vocab.get(weapon_name, 0)

    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)
    weapon_token = torch.tensor([weapon_id], device=device).unsqueeze(0)
    mask = tokens == pad_id

    model = DASHBOARD_CONTEXT.primary_model
    sae = DASHBOARD_CONTEXT.sae_model

    try:
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

        result = hidden[0, feature_id].item()
        return result

    except Exception as e:
        print(f"ERROR during compute_feature_activation: {e}")
        import traceback

        traceback.print_exc()
        return None


@dash.callback(
    Output("ablation-results-display", "children"),
    Input("run-ablation-button", "n_clicks"),
    State("ablation-primary-store", "data"),
    State("secondary-build-input", "value"),
    State("secondary-weapon-dropdown", "value"),
    State("feature-dropdown", "value"),  # Added state for selected feature
)
def run_ablation_analysis(
    n_clicks,
    primary_data,
    secondary_build_list,
    secondary_weapon_id,
    selected_feature_id,
):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return html.P(
            "Enter primary and secondary build details, then click 'Run Ablation Analysis'. Select a feature from the dropdown on the left to see its specific ablation."
        )

    if selected_feature_id is None or selected_feature_id == -1:
        return html.P(
            "Please select a feature from the dropdown on the left to see its specific ablation analysis."
        )

    if not primary_data:
        return "Please select a primary build first."

    if not secondary_build_list or len(secondary_build_list) == 0:
        return "Please select secondary ability tokens."

    # Get primary build info
    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    primary_weapon_id = primary_data.get("weapon_id_token")
    primary_ability_token_ids = primary_data.get("ability_input_tokens", [])

    if primary_weapon_id is None:
        return "Weapon ID token missing from primary build data."

    # Convert primary token IDs back to names
    primary_ability_names = [
        inv_vocab.get(token_id, f"Unknown_{token_id}")
        for token_id in primary_ability_token_ids
    ]

    # Get weapon names (use secondary weapon if selected, otherwise use primary)
    secondary_weapon_id = (
        secondary_weapon_id
        if secondary_weapon_id is not None
        else primary_weapon_id
    )

    primary_weapon_raw = inv_weapon_vocab.get(
        primary_weapon_id, f"Unknown_weapon_{primary_weapon_id}"
    )
    secondary_weapon_raw = inv_weapon_vocab.get(
        secondary_weapon_id, f"Unknown_weapon_{secondary_weapon_id}"
    )

    # Compute primary activation
    primary_activation = compute_feature_activation(
        primary_ability_names, primary_weapon_raw, selected_feature_id
    )
    if primary_activation is None:
        return "Could not compute primary activation. Check model and inputs."

    # Compute secondary activation with potentially different weapon
    secondary_activation = compute_feature_activation(
        secondary_build_list, secondary_weapon_raw, selected_feature_id
    )

    if secondary_activation is None:
        return "Could not compute secondary activation. Check model and inputs."

    # Get feature display name
    feature_name_or_id = f"Feature {selected_feature_id}"
    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        feature_name_or_id = (
            DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
                selected_feature_id
            )
        )

    # Calculate difference
    diff = secondary_activation - primary_activation

    # Get English weapon names for display
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
    primary_weapon_english = weapon_name_mapping.get(
        primary_weapon_id, primary_weapon_raw
    )
    secondary_weapon_english = weapon_name_mapping.get(
        secondary_weapon_id, secondary_weapon_raw
    )

    # Display results
    results_display = html.Div(
        [
            html.H5(f"Ablation for {feature_name_or_id}:"),
            html.P(
                f"Primary Build: {', '.join(primary_ability_names)} + {primary_weapon_english}"
            ),
            html.P(f"Primary Activation: {primary_activation:.4f}"),
            html.Hr(),
            html.P(
                f"Secondary Build: {', '.join(secondary_build_list)} + {secondary_weapon_english}"
            ),
            html.P(f"Secondary Activation: {secondary_activation:.4f}"),
            html.Hr(),
            html.P(
                f"Difference: {diff:.4f}",
                style={
                    "font-weight": "bold",
                    "color": "green" if diff > 0 else "red",
                },
            ),
            (
                html.P(
                    f"{'⚠️ Weapon changed' if primary_weapon_id != secondary_weapon_id else ''}",
                    style={"font-style": "italic", "color": "blue"},
                )
                if primary_weapon_id != secondary_weapon_id
                else html.Div()
            ),
        ]
    )

    return results_display


# Make the layout accessible for app.py
ablation_component = layout
