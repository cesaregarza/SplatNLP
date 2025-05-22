import random
from typing import Any, Optional, Tuple, Union

import dash_bootstrap_components as dbc  # Needed for dbc.Tooltip
import h5py
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html


# Helper function to format individual examples with tooltips
def format_example_with_tooltips(
    example_idx: int,
    record: pd.Series,
    inv_vocab: dict[str, str],
    inv_weapon_vocab: dict[str, str],
    feature_activation_value: float,
    model_logits: Optional[np.ndarray],
    token_activations_accessor: Optional[h5py.File],
    sae_feature_direction: Optional[np.ndarray],
    interval_num: int,
    example_in_interval_num: int,
    # json_weapon_id_to_name: Optional[dict[str, str]] = None, # Removed
) -> tuple[html.Div, list[dbc.Tooltip]]:
    if record is None:
        return html.P("Example data not found."), []

    ability_token_ids = record.get("ability_input_tokens", [])
    weapon_id_token = record.get("weapon_id_token")

    ability_names = [
        inv_vocab.get(str(tid), inv_vocab.get(tid, f"ID_{tid}"))
        for tid in ability_token_ids
    ]

    # Get the weapon name directly from inv_weapon_vocab using weapon_id_token
    # weapon_id_token is expected to be a simple ID (e.g., "0", "10", "201") from the analysis_df record.
    id_to_lookup = str(weapon_id_token) # Convert to string, assume it's the simple ID.
    
    # Perform the lookup with the new fallback format
    weapon_name = inv_weapon_vocab.get(id_to_lookup, f"UNKNOWN_WPN_ID[{id_to_lookup}]")
    # The previous logic involving conditional stripping or json_weapon_id_to_name is removed/simplified.

    top_pred_str = "N/A"  # Default
    if (
        model_logits is not None
        and isinstance(model_logits, np.ndarray)
        and model_logits.size > 0
    ):
        # Simplified top_k_predictions logic for this component
        actual_k = min(1, model_logits.size)  # Show top 1 for brevity
        if actual_k > 0:
            top_idx = np.argsort(model_logits)[-actual_k:][::-1][0]
            token_name = inv_vocab.get(
                str(top_idx), inv_vocab.get(top_idx, f"Token_ID_{top_idx}")
            )
            score = model_logits[top_idx]
            top_pred_str = f"{token_name} ({score:.2f})"

    # Prepare input abilities with tooltips
    input_abilities_spans: list[html.Span] = []
    tooltip_components: list[dbc.Tooltip] = []

    token_projections: list[str] = []
    if (
        token_activations_accessor is not None
        and sae_feature_direction is not None
    ):
        try:
            token_hidden_states_for_example = token_activations_accessor[
                str(example_idx)
            ][()]
            for k, token_id in enumerate(ability_token_ids):
                token_name_for_proj = ability_names[k]
                if token_name_for_proj in ("<PAD>", "<NULL>"):
                    continue  # Skip PAD/NULL

                if k < token_hidden_states_for_example.shape[0]:
                    token_primary_act_vector = token_hidden_states_for_example[
                        k, :
                    ]
                    projection = np.dot(
                        token_primary_act_vector, sae_feature_direction
                    )
                    token_projections.append(
                        f"{token_name_for_proj}: {projection:.3f}"
                    )
                else:
                    token_projections.append(
                        f"{token_name_for_proj}: (act_idx_err)"
                    )
        except KeyError:
            token_projections = [f"(Token acts for ex {example_idx} missing)"]
        except (
            Exception
        ):  # Broad exception for safety during HDF5 access / np.dot
            token_projections = [
                f"(Error processing token acts for ex {example_idx})"
            ]
    elif (
        token_activations_accessor is not None and sae_feature_direction is None
    ):
        token_projections = ["(SAE dir. error)"]
    else:
        token_projections = ["(Projections N/A)"]

    # Create spans and tooltips
    if not ability_names or all(
        name in ("<PAD>", "<NULL>") for name in ability_names
    ):
        input_abilities_spans.append(html.Span("N/A"))
    else:
        current_token_idx_for_projection = 0
        for i, token_name in enumerate(ability_names):
            if token_name in ("<PAD>", "<NULL>"):
                continue

            span_id = f"interval-{interval_num}-ex-{example_in_interval_num}-token-{i}"
            input_abilities_spans.append(
                html.Span(
                    token_name,
                    id=span_id,
                    style={
                        "margin-right": "5px",
                        "text-decoration": "underline dotted",
                    },
                )
            )

            tooltip_text = "Projection N/A"
            if current_token_idx_for_projection < len(token_projections):
                # Check if the current projection string is an error/placeholder or actual data
                proj_info = token_projections[current_token_idx_for_projection]
                if proj_info.startswith(f"{token_name}:"):  # It's actual data
                    tooltip_text = proj_info
                elif proj_info.startswith(
                    "("
                ):  # It's an error/placeholder message for the whole example
                    tooltip_text = proj_info
                # If token_projections is short due to an early error, this might not be specific enough.
                # For simplicity, if token_projections has one error message, all tokens show it.
                # Otherwise, it's assumed to be per-token.

            tooltip_components.append(
                dbc.Tooltip(tooltip_text, target=span_id, placement="top")
            )
            current_token_idx_for_projection += 1
            if i < len(ability_names) - 1 and ability_names[i + 1] not in (
                "<PAD>",
                "<NULL>",
            ):
                input_abilities_spans.append(", ")

    return (
        html.Div(
            [
                html.Strong(f"Weapon: {weapon_name}"),
                html.Div([html.Strong("Inputs: "), *input_abilities_spans]),
                html.P(
                    f"SAE Feature Activation: {feature_activation_value:.4f}"
                ),
                html.P(f"Top Prediction: {top_pred_str}"),
            ],
            style={
                "border": "1px solid #eee",
                "padding": "5px",
                "margin-bottom": "5px",
            },
        ),
        tooltip_components,
    )


# Main component layout
intervals_grid_component = html.Div(
    id="intervals-grid-content",
    children=[
        html.H4(
            "Subsampled Intervals Grid for SAE Feature Activations",
            className="mb-3",
        ),
        dcc.Loading(
            id="loading-intervals-grid",
            type="default",
            children=html.Div(id="intervals-grid-display"),
        ),
        html.P(id="intervals-grid-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


@callback(
    [
        Output("intervals-grid-display", "children"),
        Output("intervals-grid-error-message", "children"),
    ],
    [Input("feature-dropdown", "value")],
)
def update_intervals_grid(selected_feature_id: Optional[int]) -> tuple[list[Any], str]:
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None or DASHBOARD_CONTEXT is None:
        return [], "Select an SAE feature."

    current_run_error_messages: list[str] = []
    grid_layout_children: list[Any] = []  # For all interval sections
    all_tooltips_for_page: list[Any] = []  # For all tooltips from all examples

    all_sae_acts = DASHBOARD_CONTEXT.all_sae_hidden_activations
    analysis_df = DASHBOARD_CONTEXT.analysis_df_records  # Guaranteed DataFrame
    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab
    token_activations_accessor = getattr(
        DASHBOARD_CONTEXT, "token_activations_accessor", None
    )
    sae_model = DASHBOARD_CONTEXT.sae_model
    # json_weapon_id_to_name = getattr(
    #     DASHBOARD_CONTEXT, "json_weapon_id_to_name", None
    # ) # Removed

    if not isinstance(analysis_df, pd.DataFrame):  # Safeguard
        return [], "Error: analysis_df_records is not a DataFrame as expected."

    sae_feature_direction = None
    if (
        sae_model
        and hasattr(sae_model, "decoder")
        and hasattr(sae_model.decoder, "weight")
    ):
        sae_decoder_weights = (
            sae_model.decoder.weight.data.detach().cpu().numpy()
        )
        if 0 <= selected_feature_id < sae_decoder_weights.shape[1]:
            sae_feature_direction = sae_decoder_weights[:, selected_feature_id]
        else:
            current_run_error_messages.append(
                f"Warning: Selected feature ID {selected_feature_id} invalid for SAE decoder. Projections disabled."
            )
            sae_feature_direction = None
    elif sae_model is None:
        current_run_error_messages.append(
            "Warning: SAE model not found. Projections disabled."
        )

    if token_activations_accessor is None:
        current_run_error_messages.append(
            "Warning: Token activations HDF5 not loaded. Projections disabled."
        )

    if not (0 <= selected_feature_id < all_sae_acts.shape[1]):
        current_run_error_messages.append(
            f"Error: Feature ID {selected_feature_id} out of range for all_sae_acts."
        )
        return [], " ".join(current_run_error_messages)

    feature_activations = all_sae_acts[:, selected_feature_id]
    min_act, max_act = np.min(feature_activations), np.max(feature_activations)
    max_act = (
        max_act + 1e-6 if max_act == min_act else max_act
    )  # Avoid zero range for linspace

    num_intervals = 10
    intervals = np.linspace(min_act, max_act, num_intervals + 1)
    examples_per_interval = 2  # Reduced for brevity & to manage tooltip count

    # Create interval sections in reverse order (highest to lowest)
    for i in range(num_intervals - 1, -1, -1):  # Reverse the loop
        lower_bound, upper_bound = intervals[i], intervals[i + 1]
        condition = (feature_activations >= lower_bound) & (
            feature_activations < upper_bound
            if i < num_intervals - 1
            else feature_activations <= upper_bound
        )

        indices_in_interval = np.where(condition)[0]
        interval_section_children = [
            html.H6(
                f"Interval {num_intervals-i}: [{lower_bound:.3f}, {upper_bound:.3f}) - {len(indices_in_interval)} examples"
            )
        ]

        if len(indices_in_interval) > 0:
            num_to_sample = min(len(indices_in_interval), examples_per_interval)
            sampled_indices = random.sample(
                list(indices_in_interval), num_to_sample
            )

            for ex_in_interval_idx, original_example_idx in enumerate(
                sampled_indices
            ):
                if original_example_idx >= len(analysis_df):
                    continue  # Should not happen

                record = analysis_df.iloc[original_example_idx]
                activation_value = feature_activations[original_example_idx]
                model_logits = record.get("model_logits")

                example_div, example_tooltips = format_example_with_tooltips(
                    original_example_idx,
                    record,
                    inv_vocab,
                    inv_weapon_vocab,
                    activation_value,
                    model_logits,
                    token_activations_accessor,
                    sae_feature_direction,
                    num_intervals - i,
                    ex_in_interval_idx,  # Pass interval and example numbers for unique IDs
                    # json_weapon_id_to_name, # Removed from call
                )
                interval_section_children.append(example_div)
                all_tooltips_for_page.extend(example_tooltips)
        else:
            interval_section_children.append(
                html.P("No examples in this interval.")
            )

        grid_layout_children.append(
            html.Div(
                interval_section_children,
                className="interval-section",
                style={"margin-bottom": "15px"},
            )
        )

    # The main display div now contains both the grid layout and all tooltips
    # Tooltips are not children of their targets in Dash, they are siblings at a higher level or app level.
    # Placing them together with the grid_layout_children should work if they are correctly targeted.
    final_display_children = grid_layout_children + all_tooltips_for_page

    if (
        not grid_layout_children
    ):  # Should always have interval sections, but as a fallback
        current_run_error_messages.append(
            "Could not generate interval grid display."
        )

    return final_display_children, " ".join(current_run_error_messages)
