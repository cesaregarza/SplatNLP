import logging

import dash_ag_grid as dag
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

logger = logging.getLogger(__name__)


# Helper function for predictions (ensure this is robust)
def get_top_k_predictions(logits, inv_vocab, k=5):
    if logits is None or not isinstance(logits, np.ndarray):
        return "Logits not available"

    # Ensure logits is 1D
    if logits.ndim > 1:
        if (
            logits.shape[0] == 1 and logits.ndim == 2
        ):  # common case (1, vocab_size)
            logits = logits.squeeze(0)
        elif logits.ndim == 1:  # Already 1D
            pass
        else:  # Unexpected shape
            return f"Logits have unexpected shape {logits.shape}"

    if logits.size == 0:
        return "Logits are empty"

    actual_k = min(k, logits.size)
    if actual_k == 0:
        return "No logits to rank"

    top_k_indices = np.argsort(logits)[-actual_k:][::-1]
    predictions = []
    for idx in top_k_indices:
        token_name = inv_vocab.get(
            str(idx), inv_vocab.get(idx, f"Token_ID_{idx}")
        )
        score = logits[idx]
        predictions.append(f"{token_name} ({score:.2f})")
    return ", ".join(predictions)


# Main component layout
top_examples_component = html.Div(
    id="top-examples-content",
    children=[
        html.H4("Top Activating Examples for SAE Feature", className="mb-3"),
        dcc.Loading(
            id="loading-top-examples",
            type="default",
            children=dag.AgGrid(
                id="top-examples-grid",
                rowData=[],
                columnDefs=[],  # Will be populated by callback
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "minWidth": 150,
                },
                dashGridOptions={
                    "domLayout": "normal"  # Changed from autoHeight
                },
                style={
                    "height": "400px",
                    "width": "100%",
                },  # Added fixed height
            ),
            className="mb-2",
        ),
        html.P(id="top-examples-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


@callback(
    [
        Output("top-examples-grid", "rowData"),
        Output("top-examples-grid", "columnDefs"),
        Output("top-examples-error-message", "children"),
    ],
    [Input("feature-dropdown", "value")],
)
def update_top_examples_grid(selected_feature_id):
    from splatnlp.dashboard.app import (  # Ensure this is the correct way context is passed
        DASHBOARD_CONTEXT,
    )
    from splatnlp.preprocessing.transform.mappings import generate_maps

    logger.info(
        f"TopExamples: Received selected_feature_id: {selected_feature_id}"
    )

    # Default column definitions for early exit or error
    default_col_defs = [
        {"field": "Rank", "maxWidth": 80},
        {"field": "Weapon"},
        {
            "field": "Input Abilities",
            "wrapText": True,
            "autoHeight": True,
            "flex": 2,
        },
        {"field": "SAE Feature Activation"},
        {
            "field": "Top Predicted Abilities",
            "wrapText": True,
            "autoHeight": True,
            "flex": 2,
        },
        {"field": "Original Index", "maxWidth": 120},
    ]

    if selected_feature_id is None:
        logger.info(
            "TopExamples: selected_feature_id is None. No feature selected."
        )
        return [], default_col_defs, "Select an SAE feature."

    if DASHBOARD_CONTEXT is None:
        logger.warning("TopExamples: DASHBOARD_CONTEXT is None.")
        return (
            [],
            default_col_defs,
            "Dashboard context not available. Critical error.",
        )

    current_run_error_messages = []

    # Get data from context
    all_sae_acts = DASHBOARD_CONTEXT.all_sae_hidden_activations
    analysis_df = DASHBOARD_CONTEXT.analysis_df_records
    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab
    token_activations_accessor = getattr(
        DASHBOARD_CONTEXT, "token_activations_accessor", None
    )
    sae_model = DASHBOARD_CONTEXT.sae_model

    # Get weapon name mapping
    _, id_to_name, _ = generate_maps()

    # Validate essential data
    if not isinstance(all_sae_acts, np.ndarray):
        logger.warning(
            f"TopExamples: all_sae_acts is not a numpy array. Type: {type(all_sae_acts)}"
        )
        return (
            [],
            default_col_defs,
            "SAE activations data is missing or not in expected NumPy array format.",
        )
    else:
        logger.info(f"TopExamples: all_sae_acts shape: {all_sae_acts.shape}")

    if not isinstance(analysis_df, pd.DataFrame):
        logger.warning(
            f"TopExamples: analysis_df is not a DataFrame. Type: {type(analysis_df)}"
        )
        return (
            [],
            default_col_defs,
            "Analysis records data is not a Pandas DataFrame as expected.",
        )
    else:
        logger.info(f"TopExamples: analysis_df shape: {analysis_df.shape}")
        if analysis_df.empty:
            logger.warning("TopExamples: analysis_df is empty.")
        else:
            logger.info(
                f"TopExamples: analysis_df head:\n{analysis_df.head().to_string()}"
            )
            # Log column names to check for expected columns
            logger.info(
                f"TopExamples: analysis_df columns: {analysis_df.columns.tolist()}"
            )

    if not isinstance(inv_vocab, dict) or not isinstance(
        inv_weapon_vocab, dict
    ):
        logger.warning(
            f"TopExamples: Vocabulary data is missing or not in expected dict format. inv_vocab type: {type(inv_vocab)}, inv_weapon_vocab type: {type(inv_weapon_vocab)}"
        )
        return (
            [],
            default_col_defs,
            "Vocabulary data is missing or not in expected dict format.",
        )
    else:
        logger.info(
            f"TopExamples: inv_vocab contains {len(inv_vocab)} items. inv_weapon_vocab contains {len(inv_weapon_vocab)} items."
        )

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
                f"Warning: Selected feature ID {selected_feature_id} is invalid for SAE decoder dimensions ({sae_decoder_weights.shape[1]} features). Tooltip projections will be disabled."
            )
            sae_feature_direction = None
    elif sae_model is None:
        current_run_error_messages.append(
            "Warning: SAE model not found in context. Tooltip projections will be disabled."
        )

    if token_activations_accessor is None:
        current_run_error_messages.append(
            "Warning: Token-specific activations data not loaded (HDF5 accessor missing). Tooltips for token projections will be disabled."
        )

    # Validate selected_feature_id against all_sae_acts dimensions
    if not (0 <= selected_feature_id < all_sae_acts.shape[1]):
        logger.warning(
            f"TopExamples: selected_feature_id {selected_feature_id} is out of range for all_sae_acts with shape {all_sae_acts.shape}."
        )
        current_run_error_messages.append(
            f"Error: Feature ID {selected_feature_id} is out of range for available SAE feature activations ({all_sae_acts.shape[1]} features)."
        )
        return [], default_col_defs, " ".join(current_run_error_messages)

    feature_activations = all_sae_acts[:, selected_feature_id]
    logger.info(
        f"TopExamples: For feature {selected_feature_id}, activations min: {np.min(feature_activations):.4f}, max: {np.max(feature_activations):.4f}, mean: {np.mean(feature_activations):.4f}"
    )

    top_n = 20  # Number of top examples to show
    top_indices = np.argsort(feature_activations)[-top_n:][::-1]
    logger.info(
        f"TopExamples: Found {len(top_indices)} top indices for feature {selected_feature_id}."
    )
    if len(top_indices) > 0:
        logger.info(f"TopExamples: Top indices examples: {top_indices[:5]}")
        logger.info(
            f"TopExamples: Corresponding activations: {feature_activations[top_indices[:5]]}"
        )
    elif all_sae_acts.shape[0] > 0:  # Only log if there was data to sort
        logger.warning(
            f"TopExamples: No top indices found for feature {selected_feature_id}, despite having {all_sae_acts.shape[0]} examples. This might indicate all activations for this feature are zero or NaN."
        )

    grid_data = []
    for rank, example_idx in enumerate(top_indices):
        if example_idx >= len(
            analysis_df
        ):  # Should not happen if all_sae_acts and analysis_df are aligned
            current_run_error_messages.append(
                f"Warning: Skipping example_idx {example_idx} from top_indices as it's out of bounds for analysis_df (len {len(analysis_df)}). Data might be misaligned."
            )
            continue

        record = analysis_df.iloc[example_idx]
        ability_token_ids = record.get("ability_input_tokens", [])
        ability_names = [
            inv_vocab.get(str(tid), inv_vocab.get(tid, f"ID_{tid}"))
            for tid in ability_token_ids
        ]

        token_projection_strings = []
        if token_activations_accessor and sae_feature_direction is not None:
            try:
                # Ensure example_idx is a string for HDF5 key
                token_hidden_states_for_example = token_activations_accessor[
                    str(example_idx)
                ][()]

                for k, token_id_for_iter in enumerate(ability_token_ids):
                    token_name_for_proj = ability_names[k]
                    if token_name_for_proj in ("<PAD>", "<NULL>"):
                        token_projection_strings.append(
                            f"{token_name_for_proj}: (ignored)"
                        )
                        continue

                    if k < token_hidden_states_for_example.shape[0]:
                        token_primary_activation_vector = (
                            token_hidden_states_for_example[k, :]
                        )
                        projection = np.dot(
                            token_primary_activation_vector,
                            sae_feature_direction,
                        )
                        token_projection_strings.append(
                            f"{token_name_for_proj}: {projection:.3f}"
                        )
                    else:
                        token_projection_strings.append(
                            f"{token_name_for_proj}: (activation index out of bounds)"
                        )
            except KeyError:
                token_projection_strings = [
                    f"(Token activations for example {example_idx} not found in HDF5)"
                ]
            except Exception as e:
                token_projection_strings = [
                    f"(Error processing token activations for ex {example_idx}: {str(e)[:50]}...)"
                ]
        elif token_activations_accessor and sae_feature_direction is None:
            if not any(
                "invalid for SAE decoder" in msg
                for msg in current_run_error_messages
            ):  # Avoid duplicate general message
                token_projection_strings = [
                    "(SAE feature direction error prevented projection calculation)"
                ]
        else:
            token_projection_strings = [
                "(Token projections not available due to missing data/model)"
            ]

        # Get weapon ID and translate to name
        wid = int(record.get("weapon_id_token", -1))
        raw_wpn = inv_weapon_vocab.get(wid, f"WPN_{wid}")
        weapon_name = id_to_name.get(raw_wpn.split("_")[-1], raw_wpn)

        grid_data.append(
            {
                "Rank": rank + 1,
                "Weapon": weapon_name,
                "Input Abilities": ", ".join(ability_names),
                "SAE Feature Activation": f"{feature_activations[example_idx]:.4f}",
                "Top Predicted Abilities": get_top_k_predictions(
                    record.get("model_logits"), inv_vocab, k=5
                ),
                "ability_projections_str_list": (
                    token_projection_strings
                    if token_projection_strings
                    else ["(No projection data)"]
                ),
                "Original Index": example_idx,
            }
        )

    final_column_defs = [
        {"field": "Rank", "maxWidth": 80},
        {"field": "Weapon"},
        {
            "field": "Input Abilities",
            "wrapText": True,
            "autoHeight": True,
            "flex": 2,
            "tooltipField": "Input Abilities",  # Simple tooltip using the field value
        },
        {"field": "SAE Feature Activation"},
        {
            "field": "Top Predicted Abilities",
            "wrapText": True,
            "autoHeight": True,
            "flex": 2,
        },
        {"field": "Original Index", "maxWidth": 120},
    ]

    return grid_data, final_column_defs, " ".join(current_run_error_messages)
