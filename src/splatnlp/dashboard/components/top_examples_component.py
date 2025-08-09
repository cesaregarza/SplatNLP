import logging

import dash_ag_grid as dag
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from splatnlp.dashboard.fs_database import FSDatabase
from splatnlp.dashboard.utils.converters import generate_weapon_name_mapping

logger = logging.getLogger(__name__)

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
            "field": "Original Index",
            "headerName": "Original Index",
            "width": 120,
        },
    ]

    if selected_feature_id is None:
        return [], default_col_defs, "No feature selected."

    logger.info("TopExamples: Using database")
    db: FSDatabase = DASHBOARD_CONTEXT.db

    # Generate weapon name mapping
    weapon_name_mapping = generate_weapon_name_mapping(
        DASHBOARD_CONTEXT.inv_weapon_vocab
    )

    # Get top examples from database
    top_examples = db.get_feature_activations(selected_feature_id, limit=20)

    if len(top_examples) == 0:
        return (
            [],
            default_col_defs,
            "No top examples found for this feature.",
        )

    # Convert to grid format
    grid_data = []
    for i, example in enumerate(top_examples.to_dicts(), 1):
        # Get weapon name from weapon_id
        weapon_name = weapon_name_mapping.get(
            int(example.get("weapon_id", 0)),
            f"Weapon_{example.get('weapon_id', 'unknown')}",
        )

        # Get ability tags
        ability_tags = []
        if (
            "ability_input_tokens" in example
            and example["ability_input_tokens"] is not None
        ):
            try:
                # Convert tags to names using vocabulary
                ability_tags = [
                    DASHBOARD_CONTEXT.inv_vocab.get(int(tag), f"Token_{tag}")
                    for tag in example["ability_input_tokens"]
                ]
            except Exception as e:
                logger.warning(f"Error processing ability tags: {e}")
                ability_tags = ["Error processing tags"]

        grid_data.append(
            {
                "Rank": i,
                "Weapon": weapon_name,
                "Input Abilities": ", ".join(ability_tags),
                "SAE Feature Activation": f"{example.get('activation', 0):.4f}",
                "Original Index": example.get("index", "N/A"),
            }
        )

    return grid_data, default_col_defs, ""
