import json
import logging

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

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
                    "domLayout": "normal",
                    "rowSelection": "multiple",
                },
                style={
                    "height": "400px",
                    "width": "100%",
                },
            ),
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Add Selected to Sweep Queue",
                        id="add-examples-to-sweep-queue-button",
                        color="secondary",
                        size="sm",
                        className="me-2",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    html.Span(
                        id="add-to-sweep-status",
                        className="text-muted small align-self-center",
                    ),
                    width="auto",
                ),
            ],
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
    [
        Input("feature-dropdown", "value"),
        Input("active-tab-store", "data"),
    ],
)
def update_top_examples_grid(selected_feature_id, active_tab):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Lazy loading: skip if tab is not active
    if active_tab != "tab-examples":
        return no_update, no_update, no_update

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
        # Hidden columns for build data (used by Add to Sweep Queue)
        {"field": "weapon_id", "hide": True},
        {"field": "ability_tokens_json", "hide": True},
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
                ability_tags = [
                    DASHBOARD_CONTEXT.inv_vocab.get(int(tag), f"Token_{tag}")
                    for tag in example["ability_input_tokens"]
                ]
            except (TypeError, ValueError) as e:
                logger.warning(f"Error processing ability tags: {e}")
                ability_tags = ["Error processing tags"]

        grid_data.append(
            {
                "Rank": i,
                "Weapon": weapon_name,
                "Input Abilities": ", ".join(ability_tags),
                "SAE Feature Activation": f"{example.get('activation', 0):.4f}",
                "Original Index": example.get("index", "N/A"),
                # Hidden data for sweep queue
                "weapon_id": example.get("weapon_id"),
                "ability_tokens_json": json.dumps(ability_tags),
            }
        )

    return grid_data, default_col_defs, ""


@callback(
    Output("full-sweep-build-queue", "data", allow_duplicate=True),
    Output("add-to-sweep-status", "children"),
    Input("add-examples-to-sweep-queue-button", "n_clicks"),
    State("top-examples-grid", "selectedRows"),
    State("full-sweep-build-queue", "data"),
    prevent_initial_call=True,
)
def add_selected_examples_to_sweep_queue(n_clicks, selected_rows, current_queue):
    """Add selected examples from the grid to the full sweep queue."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return dash.no_update, ""

    if not selected_rows or len(selected_rows) == 0:
        return dash.no_update, "No rows selected"

    # Initialize queue if None
    if current_queue is None:
        current_queue = []

    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)

    added_count = 0
    for row in selected_rows:
        weapon_id = row.get("weapon_id")
        ability_tokens_json = row.get("ability_tokens_json", "[]")

        if weapon_id is None:
            continue

        try:
            ability_tokens = json.loads(ability_tokens_json)
        except json.JSONDecodeError:
            ability_tokens = []

        # Get weapon name
        weapon_name = weapon_name_mapping.get(weapon_id, row.get("Weapon", "Unknown"))

        # Create description
        abilities_str = ", ".join(ability_tokens[:3])
        if len(ability_tokens) > 3:
            abilities_str += f"... (+{len(ability_tokens) - 3})"
        description = f"{abilities_str} + {weapon_name}"

        # Add to queue
        new_build = {
            "weapon_id": weapon_id,
            "weapon_name": weapon_name,
            "ability_tokens": ability_tokens,
            "description": description,
        }
        current_queue.append(new_build)
        added_count += 1

    status_msg = f"Added {added_count} build(s) to queue"
    return current_queue, status_msg
