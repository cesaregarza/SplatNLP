import json
import logging
from typing import Optional

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

    logger.info(
        f"TopExamples: Received selected_feature_id: {selected_feature_id}"
    )

    # Default column definitions
    default_col_defs = [
        {"field": "Rank", "headerName": "Rank", "width": 80},
        {"field": "Weapon", "headerName": "Weapon", "width": 150},
        {
            "field": "Input Abilities",
            "headerName": "Input Abilities",
            "width": 200,
        },
        {
            "field": "SAE Feature Activation",
            "headerName": "SAE Feature Activation",
            "width": 150,
        },
        {
            "field": "Top Predicted Abilities",
            "headerName": "Top Predicted Abilities",
            "width": 200,
        },
        {
            "field": "Original Index",
            "headerName": "Original Index",
            "width": 120,
        },
    ]

    if not selected_feature_id:
        return [], default_col_defs, "No feature selected."

    # Use database-backed context if available
    if hasattr(DASHBOARD_CONTEXT, "db"):
        logger.info("TopExamples: Using filesystem database")
        examples = DASHBOARD_CONTEXT.db.get_top_examples(selected_feature_id)
        if not examples:
            return [], default_col_defs, "No examples found for this feature."

        # Convert to grid format
        grid_data = []
        for i, ex in enumerate(examples, 1):
            # Handle array format from filesystem database
            if isinstance(ex.get("ability_input_tokens"), str):
                try:
                    ex["ability_input_tokens"] = json.loads(
                        ex["ability_input_tokens"]
                    )
                except json.JSONDecodeError:
                    ex["ability_input_tokens"] = []

            grid_data.append(
                {
                    "Rank": i,
                    "Weapon": ex.get("weapon_name", "N/A"),
                    "Input Abilities": ", ".join(
                        ex.get("ability_input_tokens", [])
                    )
                    or "N/A",
                    "SAE Feature Activation": f"{ex.get('activation_value', 0):.4f}",
                    "Top Predicted Abilities": ex.get(
                        "top_predicted_abilities_str", "N/A"
                    ),
                    "Original Index": ex["index"],
                }
            )

        return grid_data, default_col_defs, ""

    logger.warning("TopExamples: No database context available.")
    return [], default_col_defs, "Error: No database context available."


def update_top_examples(feature_id: Optional[int]):
    """Update top examples when feature selection changes."""
    if feature_id is None or feature_id == -1:
        return html.Div("Select a feature to view top examples")

    try:
        logger.info("TopExamples: Using filesystem database")
        examples = DASHBOARD_CONTEXT.db.get_top_examples(feature_id)
        if not examples:
            return html.Div("No examples found for this feature")

        # Create table rows
        rows = []
        for ex in examples:
            # Handle array format from filesystem database
            if isinstance(ex.get("ability_input_tokens"), str):
                try:
                    ex["ability_input_tokens"] = json.loads(
                        ex["ability_input_tokens"]
                    )
                except json.JSONDecodeError:
                    ex["ability_input_tokens"] = []

            rows.append(
                html.Tr(
                    [
                        html.Td(ex.get("weapon_name", "N/A")),
                        html.Td(
                            ", ".join(ex.get("ability_input_tokens", []))
                            or "N/A"
                        ),
                        html.Td(ex.get("input_abilities_str", "N/A")),
                        html.Td(ex.get("top_predicted_abilities_str", "N/A")),
                        html.Td(f"{ex.get('activation_value', 0):.4f}"),
                    ]
                )
            )

        return html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Weapon"),
                            html.Th("Input Tokens"),
                            html.Th("Input Abilities"),
                            html.Th("Predicted Abilities"),
                            html.Th("Activation"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            className="table table-striped",
        )
    except Exception as e:
        logger.error(f"Error updating top examples: {e}", exc_info=True)
        return html.Div(f"Error loading examples: {str(e)}")
