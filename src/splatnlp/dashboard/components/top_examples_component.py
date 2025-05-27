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
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
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

    # Use database-backed context if available
    if hasattr(DASHBOARD_CONTEXT, 'db_context'):
        logger.info("TopExamples: Using database-backed context")
        db_context = DASHBOARD_CONTEXT.db_context
        
        try:
            # Get top examples from database
            top_examples = db_context.get_top_examples(selected_feature_id, limit=20)
            
            if not top_examples:
                return [], default_col_defs, "No top examples found for this feature."
            
            # Convert to grid format
            grid_data = []
            for example in top_examples:
                grid_data.append({
                    "Rank": example['rank'],
                    "Weapon": example['weapon_name'],
                    "Input Abilities": example['input_abilities_str'],
                    "SAE Feature Activation": f"{example['activation_value']:.4f}",
                    "Top Predicted Abilities": example['top_predicted_abilities_str'],
                    "Original Index": example['example_id'],
                })
            
            return grid_data, default_col_defs, ""
            
        except Exception as e:
            logger.error(f"TopExamples: Database error: {e}")
            return [], default_col_defs, f"Database error: {str(e)}"
    
    # The block below using DASHBOARD_CONTEXT.all_sae_hidden_activations and 
    # DASHBOARD_CONTEXT.analysis_df_records is the fallback/old method.
    # This should be removed as per the refactoring task to rely solely on db_context
    # when it's available (which it should be for the 'run' command).
    
    # If db_context was not available, the function would have already returned above.
    # Thus, this part of the code should ideally not be reached if db_context is present.
    # If precomputed_analytics were an alternative data source for other commands,
    # that would need explicit handling based on args or context state.
    # For the 'run' command, precomputed_analytics is None.

    logger.warning("TopExamples: Reached fallback logic that relies on in-memory precomputed_analytics. This should not happen if db_context is used.")
    return [], default_col_defs, "Error: Fallback to in-memory data not supported when db_context is primary."
