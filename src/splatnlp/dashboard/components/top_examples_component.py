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
        logger.info("TopExamples: Using DuckDB database")
        db = DASHBOARD_CONTEXT.db

        try:
            # Get top examples from database
            activations_df = db.get_feature_activations(
                selected_feature_id, limit=20
            )

            if activations_df.empty:
                return (
                    [],
                    default_col_defs,
                    "No top examples found for this feature.",
                )

            # Convert to grid format
            grid_data = []
            for i, (_, row) in enumerate(activations_df.iterrows(), 1):
                # Get weapon name from weapon_id
                weapon_name = f"Weapon_{row['weapon_id']}"
                if hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab"):
                    weapon_name = DASHBOARD_CONTEXT.inv_weapon_vocab.get(
                        str(row["weapon_id"]), weapon_name
                    )

                # Get ability tags
                ability_tags = []
                if "ability_tags" in row and row["ability_tags"] is not None:
                    try:
                        # Handle array format from DuckDB
                        tags = row["ability_tags"]
                        if isinstance(tags, str):
                            # Handle string format (comma-separated)
                            tags = [
                                int(t.strip())
                                for t in tags.strip("[]").split(",")
                                if t.strip()
                            ]
                        elif isinstance(tags, list):
                            # Already in list format
                            tags = [int(t) for t in tags if t is not None]

                        # Convert tags to names using vocabulary
                        if hasattr(DASHBOARD_CONTEXT, "inv_vocab"):
                            ability_tags = [
                                DASHBOARD_CONTEXT.inv_vocab.get(
                                    str(tag), f"Token_{tag}"
                                )
                                for tag in tags
                            ]
                        else:
                            ability_tags = [f"Token_{tag}" for tag in tags]
                    except Exception as e:
                        logger.warning(f"Error processing ability tags: {e}")
                        ability_tags = ["Error processing tags"]

                grid_data.append(
                    {
                        "Rank": i,
                        "Weapon": weapon_name,
                        "Input Abilities": ", ".join(ability_tags),
                        "SAE Feature Activation": f"{row['activation']:.4f}",
                        "Top Predicted Abilities": "N/A",  # Not available in new format
                        "Original Index": row["index"],
                    }
                )

            return grid_data, default_col_defs, ""

        except Exception as e:
            logger.error(f"TopExamples: Database error: {e}")
            return [], default_col_defs, f"Database error: {str(e)}"

    logger.warning("TopExamples: No database context available.")
    return [], default_col_defs, "Error: No database context available."
