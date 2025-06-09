import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import json # For potentially displaying dicts or if data is stored as JSON string
import torch # Import PyTorch
# Removed: from splatnlp.dashboard.app import DASHBOARD_CONTEXT

# Define the layout for the Ablation tab
layout = html.Div(
    [
        dcc.Store(id='ablation-secondary-store'), # Store for modified/secondary build data
        html.H3("Ablation Analysis"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Primary Build Details"),
                        html.Div(
                            id='primary-build-display',
                            children="No primary build selected.",
                            className="mb-3 p-2 border rounded bg-light",
                            style={'minHeight': '100px'}
                        ),
                    ],
                    md=6
                ),
                dbc.Col(
                    [
                        html.H4("Secondary Build Input (Modify Abilities)"),
                        dbc.Textarea(
                            id='secondary-build-input',
                            placeholder="Modify ability tokens here, comma-separated...",
                            className="mb-2",
                            rows=3,
                        ),
                        dbc.Button(
                            "Run Ablation Analysis",
                            id='run-ablation-button',
                            color="success",
                            className="mt-2"
                        ),
                    ],
                    md=6
                ),
            ]
        ),
        html.Hr(),
        html.H4("Ablation Results"),
        html.Div(
            id='ablation-results-display',
            children="Ablation results will appear here.", # Placeholder
            className="mt-3 p-2 border rounded"
        )
    ]
)

# Callback to display primary build details and pre-fill secondary input
@dash.callback(
    [Output('primary-build-display', 'children'),
     Output('secondary-build-input', 'value')],
    Input('ablation-primary-store', 'data')
)
def display_primary_build(primary_data):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT # Moved import here
    if primary_data:
        inv_vocab = DASHBOARD_CONTEXT.inv_vocab if hasattr(DASHBOARD_CONTEXT, 'inv_vocab') else {}
        weapon_id = primary_data.get('weapon_id_token', 'N/A')
        ability_token_ids = primary_data.get('ability_input_tokens', [])
        activation = primary_data.get('activation', 'N/A')

        ability_names = [str(inv_vocab.get(token_id, token_id)) for token_id in ability_token_ids]
        abilities_display_str = ", ".join(ability_names) if ability_names else "None"
        # Use the same names for pre-filling the secondary input
        abilities_input_str = ", ".join(ability_names) if ability_names else ""


        display_children = [
            html.P(f"Weapon ID: {weapon_id}"), # Assuming weapon_id is already a name or appropriate ID
            html.P(f"Abilities: {abilities_display_str}"),
            html.P(f"Original Activation: {activation:.4f}" if isinstance(activation, float) else f"Original Activation: {activation}")
        ]
        return display_children, abilities_input_str
    else:
        return "No primary build selected.", ""

# Callback to update the secondary store from the textarea
@dash.callback(
    Output('ablation-secondary-store', 'data'),
    Input('secondary-build-input', 'value')
)
def update_secondary_store(secondary_build_string):
    # Store the raw string. Processing/parsing will happen when "Run Ablation" is clicked.
    if secondary_build_string is None:
        return dash.no_update # Or {} or None depending on how you want to handle empty input
    return {'ability_tokens_string': secondary_build_string}

# Helper function to get SAE activations
@torch.no_grad()
def get_sae_activations(primary_model, sae_model, ability_token_ids: list[int], weapon_token_id: int, pad_token_id: int, device: str) -> torch.Tensor | None:
    """
    Computes SAE activations for a given set of ability and weapon tokens.
    """
    if primary_model is None or sae_model is None:
        print("Warning: Primary model or SAE model is None in get_sae_activations.")
        return None

    try:
        # Prepare inputs
        ability_tokens_tensor = torch.tensor([ability_token_ids], dtype=torch.long, device=device) # Shape: (1, seq_len)
        weapon_token_tensor = torch.tensor([[weapon_token_id]], dtype=torch.long, device=device) # Shape: (1, 1)

        # Create key_padding_mask: True for padded tokens
        key_padding_mask = (ability_tokens_tensor == pad_token_id)

        # Replicate primary_model processing
        ability_embeddings = primary_model.token_embed(ability_tokens_tensor) # (1, seq_len, d_model)
        weapon_embeddings = primary_model.weapon_embed(weapon_token_tensor)   # (1, 1, d_model)

        # Expand weapon_embeddings to match ability_embeddings sequence length
        # The original model might have specific broadcasting rules or direct additions if shapes are compatible.
        # Assuming weapon_embeddings should be added to each token embedding in the sequence.
        embeddings = ability_embeddings + weapon_embeddings.expand_as(ability_embeddings)

        x = primary_model.input_proj(embeddings) # Project to transformer d_model if different

        # Transformer layers
        for layer in primary_model.transformer_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask) # Pass mask here

        # Masked mean
        masked_mean_output = primary_model.masked_mean(x, key_padding_mask.unsqueeze(-1)) # Unsqueeze mask for broadcasting

        # Get SAE activations
        _, h_post = sae_model.encode(masked_mean_output) # h_post shape (1, d_sae)

        return h_post.squeeze() # Shape: (d_sae)
    except Exception as e:
        print(f"Error during get_sae_activations: {e}")
        return None

@dash.callback(
    Output('ablation-results-display', 'children'),
    Input('run-ablation-button', 'n_clicks'),
    State('ablation-primary-store', 'data'),
    State('secondary-build-input', 'value'),
    State('feature-dropdown', 'value')  # Added state for selected feature
)
def run_ablation_analysis(n_clicks, primary_data, secondary_build_string, selected_feature_id): # Added argument
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT # Moved import here
    if not n_clicks:
        # Return a default message if the button hasn't been clicked yet in the session
        return html.P("Enter primary and secondary build details, then click 'Run Ablation Analysis'. Select a feature from the dropdown on the left to see its specific ablation.")

    # Fetch models and config from DASHBOARD_CONTEXT
    primary_model = getattr(DASHBOARD_CONTEXT, 'primary_model', None)
    sae_model = getattr(DASHBOARD_CONTEXT, 'sae_model', None)
    vocab = getattr(DASHBOARD_CONTEXT, 'vocab', {})
    # inv_vocab = getattr(DASHBOARD_CONTEXT, 'inv_vocab', {}) # Not directly needed here but good to be aware of
    pad_token_id = getattr(DASHBOARD_CONTEXT, 'pad_token_id', None)
    device = getattr(DASHBOARD_CONTEXT, 'device', 'cpu') # Default to CPU if not specified

    if primary_model is None or sae_model is None:
        return html.Div("Models not loaded. Ablation analysis unavailable.", style={'color': 'red'})

    if pad_token_id is None:
        return html.Div("PAD token ID not configured in dashboard context. This might be due to the '<PAD>' token missing from the vocabulary file used at startup. Ablation analysis unavailable.", style={'color': 'red'})

    if not primary_data:
        return "Please select a primary build first."

    if not secondary_build_string or not secondary_build_string.strip():
        return "Please provide secondary ability tokens."

    weapon_id_token = primary_data.get('weapon_id_token')
    primary_ability_token_ids = primary_data.get('ability_input_tokens', [])

    if weapon_id_token is None:
        return "Weapon ID token missing from primary build data."

    # Get Primary Activations
    primary_acts = get_sae_activations(
        primary_model, sae_model, primary_ability_token_ids, weapon_id_token, pad_token_id, device
    )

    if primary_acts is None:
        return "Could not compute primary activations. Check model and inputs."

    # Parse secondary build string and convert to token IDs
    secondary_ability_names = [name.strip() for name in secondary_build_string.split(',') if name.strip()]
    secondary_ability_token_ids = []
    unknown_ability_names = []
    warning_message = ""

    for name in secondary_ability_names:
        token_id = vocab.get(name)
        if token_id is not None:
            secondary_ability_token_ids.append(token_id)
        else:
            unknown_ability_names.append(name)

    if unknown_ability_names:
        warning_message = f"Warning: The following ability names were not found in vocabulary and will be skipped: {', '.join(unknown_ability_names)}"

    if not secondary_ability_token_ids: # If all names were unknown or input was effectively empty after parsing
        return html.Div([
            html.P(warning_message, style={'color': 'orange'}) if warning_message else "",
            html.P("No valid secondary ability tokens to process after parsing.", style={'color': 'red'})
        ])

    # Get Secondary Activations
    secondary_acts = get_sae_activations(
        primary_model, sae_model, secondary_ability_token_ids, weapon_id_token, pad_token_id, device
    )

    if secondary_acts is None:
        return html.Div([
            html.P(warning_message, style={'color': 'orange'}) if warning_message else "",
            html.P("Could not compute secondary activations. Check model and inputs.", style={'color': 'red'})
        ])

    # Focused Ablation Logic
    if selected_feature_id is None or selected_feature_id == -1: # Often -1 is used for placeholder/unselected
        results_display = html.Div([
            html.P("Please select a feature from the dropdown on the left to see its specific ablation analysis.")
        ])
    else:
        try:
            # Ensure selected_feature_id is an integer
            feature_id = int(selected_feature_id)

            # Validate feature_id bounds
            if not (0 <= feature_id < primary_acts.shape[0]):
                results_display = html.Div(
                    f"Error: Selected feature ID {feature_id} is out of bounds for the activation tensor (max index: {primary_acts.shape[0] - 1}).",
                    style={'color': 'red'}
                )
            else:
                feature_name_or_id = f"Feature {feature_id}"
                if hasattr(DASHBOARD_CONTEXT, 'feature_labels_manager') and DASHBOARD_CONTEXT.feature_labels_manager:
                    # This might return None or the ID itself if no custom label
                    # get_display_name often returns the ID if no name is set.
                    feature_name_or_id = DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(feature_id)

                primary_val = primary_acts[feature_id].item()
                secondary_val = secondary_acts[feature_id].item()
                diff = secondary_val - primary_val

                results_display = html.Div([
                    html.H5(f"Ablation for {feature_name_or_id}:"),
                    html.P(f"Primary Activation: {primary_val:.4f}"),
                    html.P(f"Secondary Activation: {secondary_val:.4f}"),
                    html.P(f"Difference: {diff:.4f}")
                ])
        except ValueError:
            results_display = html.Div(
                f"Error: Invalid feature ID format: {selected_feature_id}.",
                style={'color': 'red'}
            )
        except Exception as e: # Catch any other unexpected errors during focused display
            results_display = html.Div(
                f"An error occurred while processing selected feature {selected_feature_id}: {str(e)}",
                style={'color': 'red'}
            )

    if warning_message:
        # Prepend warning to the results display
        return html.Div([html.P(warning_message, style={'color': 'orange'}), results_display])

    return results_display

# Make the layout accessible for app.py
ablation_component = layout
