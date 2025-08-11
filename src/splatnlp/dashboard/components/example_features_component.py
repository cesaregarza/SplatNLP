import json
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import (
    Input,
    Output,
    State,
    callback,
    callback_context,
    dcc,
    html,
    no_update,
)
from dash.dependencies import MATCH

from splatnlp.dashboard.efficient_fs_database import EfficientFSDatabase
from splatnlp.dashboard.fs_database import FSDatabase
from splatnlp.dashboard.utils.converters import generate_weapon_name_mapping

logger = logging.getLogger(__name__)
logger.info("example_features_component module loaded")

# Configuration
TOP_K_EXAMPLES = 25  # Number of top examples to show
TOP_K_FEATURES = 30  # Number of top features per example to analyze
TOP_K_COACTIVATING = 20  # Number of top co-activating features to display


def create_example_card(
    example_data: Dict,
    example_rank: int,
    card_id: str = None,
    highlighted: bool = False,
) -> dbc.Card:
    """Create a card displaying a single example."""
    weapon_name = example_data.get("weapon_name", "Unknown")
    abilities = example_data.get("abilities", [])
    activation_value = example_data.get("activation", 0.0)
    original_idx = example_data.get("original_idx", -1)

    # Format abilities display
    abilities_str = " | ".join(abilities) if abilities else "No abilities"

    # Add highlight style if needed
    card_style = {"fontSize": "0.9rem"}
    if highlighted:
        card_style.update(
            {
                "border": "2px solid #007bff",
                "boxShadow": "0 0 10px rgba(0,123,255,0.3)",
            }
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        [
                            html.Span(
                                f"Example #{example_rank}", className="fw-bold"
                            ),
                            html.Span(
                                f"Index: {original_idx}",
                                className="text-muted ms-2 small",
                            ),
                        ],
                        className="d-flex justify-content-between",
                    )
                ]
            ),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Strong("Weapon: "),
                            html.Span(weapon_name),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Strong("Abilities: "),
                            html.Div(
                                abilities_str,
                                className="text-wrap small",
                                style={"maxWidth": "100%"},
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Strong("Activation: "),
                            html.Span(
                                f"{activation_value:.4f}",
                                className="badge bg-primary",
                            ),
                        ],
                    ),
                ]
            ),
        ],
        className="mb-2",
        style=card_style,
        id=card_id if card_id else f"example-card-{example_rank}",
    )


# Main component layout
example_features_component = html.Div(
    id="example-features-content",
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Co-Activation Analysis", className="mb-3"),
                        html.P(
                            "Analyze which features commonly activate together with the selected feature.",
                            className="text-muted",
                        ),
                    ],
                    width=12,
                ),
            ]
        ),
        dcc.Store(
            id="example-indices-store", storage_type="memory"
        ),  # Store example indices
        dcc.Store(
            id="coactivation-data-store", storage_type="memory"
        ),  # Store co-activation analysis
        dcc.Store(
            id="example-features-map", storage_type="memory"
        ),  # Store which examples have which features
        dcc.Store(
            id="highlighted-feature", storage_type="memory"
        ),  # Store currently highlighted feature
        # Control section
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P(
                                            "Load features for all top examples to analyze co-activation patterns:",
                                            className="mb-2",
                                        ),
                                    ],
                                    width=8,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Analyze Co-Activations",
                                            id="analyze-coactivations-btn",
                                            color="primary",
                                            size="md",
                                            className="w-100",
                                            n_clicks=0,
                                        ),
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                    ]
                )
            ],
            className="mb-3",
        ),
        # Results section
        dbc.Row(
            [
                # Left column: Examples
                dbc.Col(
                    [
                        html.H5("Top Activating Examples", className="mb-3"),
                        dcc.Loading(
                            id="loading-example-cards",
                            type="default",
                            children=html.Div(id="example-cards-container"),
                        ),
                    ],
                    width=4,
                ),
                # Right column: Co-activation analysis
                dbc.Col(
                    [
                        html.H5("Co-Activating Features", className="mb-3"),
                        dcc.Loading(
                            id="loading-coactivation-analysis",
                            type="default",
                            children=html.Div(
                                id="coactivation-analysis-container"
                            ),
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        html.Div(id="example-features-error", style={"color": "red"}),
    ],
    className="p-3",
)


def update_example_cards(selected_feature_id):
    """Load top examples for the selected feature and create cards."""
    logger.info(
        f"[example_features] update_example_cards implementation called with {selected_feature_id}"
    )

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    logger.info(
        f"[example_features] DASHBOARD_CONTEXT imported, has db: {hasattr(DASHBOARD_CONTEXT, 'db')}"
    )

    if selected_feature_id is None:
        logger.info("[example_features] No feature selected, returning empty")
        return [], {}, "No feature selected."

    try:
        db = DASHBOARD_CONTEXT.db
        logger.info(f"[example_features] Got db of type: {type(db)}")

        # Generate weapon name mapping
        weapon_name_mapping = generate_weapon_name_mapping(
            DASHBOARD_CONTEXT.inv_weapon_vocab
        )

        # Get top examples
        if isinstance(db, EfficientFSDatabase):
            # Use efficient database method
            examples_df = db.get_top_examples(
                selected_feature_id, limit=TOP_K_EXAMPLES
            )

            if examples_df is None or len(examples_df) == 0:
                return (
                    [],
                    {},
                    f"No examples found for feature {selected_feature_id}",
                )

            logger.info(
                f"[example_features] DataFrame columns: {examples_df.columns.tolist()}"
            )
            logger.info(
                f"[example_features] DataFrame shape: {examples_df.shape}"
            )

            # Store example info for later feature retrieval
            # We need both the global index and the batch/sample info
            example_info = []
            example_full_data = []  # Store full data for card recreation
            for _, row in examples_df.iterrows():
                example_info.append(
                    {
                        "global_index": row["index"],
                        "batch_id": row.get("batch_id"),
                        "sample_id": row.get("sample_id"),
                    }
                )

            # Create cards for each example
            cards = []
            for rank, (idx, row) in enumerate(examples_df.iterrows(), 1):
                # Parse abilities from token indices
                # The column is named 'ability_input_tokens' in the efficient database
                abilities = []
                token_list = row.get(
                    "ability_input_tokens", row.get("token_indices", [])
                )
                for token_idx in token_list:
                    if token_idx in DASHBOARD_CONTEXT.inv_vocab:
                        token = DASHBOARD_CONTEXT.inv_vocab[token_idx]
                        if (
                            token != "<PAD>"
                            and token != "<NULL>"
                            and not token.startswith("WPN_")
                        ):
                            abilities.append(token)

                # Get weapon from weapon_id_token or weapon_id
                weapon_id = row.get(
                    "weapon_id_token", row.get("weapon_id", None)
                )

                # Look up weapon name directly using the integer ID
                weapon_name = weapon_name_mapping.get(
                    int(weapon_id) if weapon_id is not None else 0,
                    (
                        f"Weapon_{weapon_id}"
                        if weapon_id is not None
                        else "Unknown"
                    ),
                )

                example_data = {
                    "weapon_name": weapon_name,
                    "abilities": abilities,
                    "activation": row.get(
                        "activation", row.get("activation_value", 0.0)
                    ),
                    "original_idx": row["index"],
                }

                example_full_data.append(example_data)  # Store for later use
                cards.append(create_example_card(example_data, rank))

            return (
                cards,
                {
                    "examples": example_info,
                    "full_data": example_full_data,
                    "feature_id": selected_feature_id,
                },
                "",
            )

        else:
            # Use legacy database method
            top_examples = db.get_feature_activations(
                selected_feature_id, limit=TOP_K_EXAMPLES
            )

            if len(top_examples) == 0:
                return (
                    [],
                    {},
                    f"No examples found for feature {selected_feature_id}",
                )

            example_info = []
            example_full_data = []
            cards = []

            for rank, (activation_value, original_idx) in enumerate(
                top_examples, 1
            ):
                example_info.append(
                    {
                        "global_index": original_idx,
                        "batch_id": None,
                        "sample_id": None,
                    }
                )

                # Get example data from metadata
                example_row = DASHBOARD_CONTEXT.metadata[
                    DASHBOARD_CONTEXT.metadata["original_index"] == original_idx
                ].iloc[0]

                # Parse abilities
                abilities = []
                for col in ["ability_0", "ability_1", "ability_2"]:
                    if col in example_row:
                        ability = example_row[col]
                        if pd.notna(ability) and ability != "<PAD>":
                            abilities.append(ability)

                weapon_name = weapon_name_mapping.get(
                    example_row.get("weapon", "Unknown"), "Unknown"
                )

                example_data = {
                    "weapon_name": weapon_name,
                    "abilities": abilities,
                    "activation": activation_value,
                    "original_idx": original_idx,
                }

                example_full_data.append(example_data)
                cards.append(create_example_card(example_data, rank))

            return (
                cards,
                {
                    "examples": example_info,
                    "full_data": example_full_data,
                    "feature_id": selected_feature_id,
                },
                "",
            )

    except Exception as e:
        logger.error(f"Error loading examples: {e}")
        return [], {}, f"Error loading examples: {str(e)}"


def analyze_coactivations(n_clicks, example_indices_data):
    """Analyze co-activating features for all examples.

    Returns:
        tuple: (html content, example_features_map for storage)
    """
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if n_clicks == 0 or n_clicks is None:
        return (
            html.Div(
                "Click 'Analyze Co-Activations' to see analysis",
                className="text-muted",
            ),
            {},
        )

    # Handle both old and new data formats
    if not example_indices_data:
        return (
            html.Div(
                "Error: No example data available", className="text-danger"
            ),
            {},
        )

    if "examples" in example_indices_data:
        # New format with batch/sample info
        example_info = example_indices_data["examples"]
    elif "indices" in example_indices_data:
        # Old format - convert to new
        example_info = [
            {"global_index": idx, "batch_id": None, "sample_id": None}
            for idx in example_indices_data["indices"]
        ]
    else:
        return (
            html.Div(
                "Error: Invalid example data format", className="text-danger"
            ),
            {},
        )

    selected_feature_id = example_indices_data.get("feature_id")

    logger.info(
        f"[coactivation] Analyzing {len(example_info)} examples for feature {selected_feature_id}"
    )

    try:
        db = DASHBOARD_CONTEXT.db

        if isinstance(db, EfficientFSDatabase):
            # Collect all features that activate for these examples
            feature_counter = Counter()
            feature_activations = {}  # Store activation values for averaging
            example_features_map = (
                {}
            )  # Map of feature_id -> list of example indices that have it

            for i, ex_info in enumerate(example_info):
                # Get top features for this example
                # If we have batch/sample info, use a more efficient method
                if (
                    ex_info.get("batch_id") is not None
                    and ex_info.get("sample_id") is not None
                ):
                    activations = db.get_example_activations_by_batch(
                        ex_info["batch_id"],
                        ex_info["sample_id"],
                        top_k=TOP_K_FEATURES,
                    )
                else:
                    # Fall back to global index
                    activations = db.get_example_activations(
                        ex_info["global_index"], top_k=TOP_K_FEATURES
                    )

                logger.info(
                    f"[coactivation] Example {i+1}/{len(example_info)}: idx={ex_info['global_index']}, found {len(activations) if activations else 0} activations"
                )

                if activations:
                    for feat_id, activation_val in activations:
                        # Skip the selected feature itself
                        if feat_id != selected_feature_id:
                            feature_counter[feat_id] += 1
                            if feat_id not in feature_activations:
                                feature_activations[feat_id] = []
                                example_features_map[feat_id] = []
                            feature_activations[feat_id].append(activation_val)
                            example_features_map[feat_id].append(
                                i
                            )  # Store example index
                else:
                    logger.warning(
                        f"[coactivation] No activations found for example {ex_info}"
                    )

            if not feature_counter:
                return (
                    html.Div(
                        "No co-activating features found",
                        className="text-muted",
                    ),
                    {},
                )

            # Get top co-activating features
            top_coactivating = feature_counter.most_common(TOP_K_COACTIVATING)

            # Create analysis display
            feature_rows = []
            for feat_id, count in top_coactivating:
                # Calculate average activation
                avg_activation = np.mean(feature_activations[feat_id])

                # Get feature label if available
                label = ""
                if hasattr(DASHBOARD_CONTEXT, "feature_labels_manager"):
                    label_data = (
                        DASHBOARD_CONTEXT.feature_labels_manager.get_label(
                            feat_id
                        )
                    )
                    if label_data:
                        label = label_data.get("label", "")

                # Calculate co-occurrence percentage
                co_occurrence_pct = (count / len(example_info)) * 100

                # Which examples contain this feature
                example_indices_str = ", ".join(
                    [str(i + 1) for i in example_features_map[feat_id][:5]]
                )
                if len(example_features_map[feat_id]) > 5:
                    example_indices_str += (
                        f"... (+{len(example_features_map[feat_id])-5} more)"
                    )

                feature_rows.append(
                    html.Tr(
                        [
                            html.Td(f"F{feat_id}"),
                            html.Td(f"{count}/{len(example_info)}"),
                            html.Td(f"{co_occurrence_pct:.1f}%"),
                            html.Td(f"{avg_activation:.4f}"),
                            html.Td(label or "-", className="small"),
                            html.Td(
                                example_indices_str,
                                className="small text-muted",
                            ),
                        ],
                        id=f"feature-row-{feat_id}",
                    )
                )

            return (
                html.Div(
                    [
                        html.P(
                            f"Found {len(feature_counter)} unique co-activating features across {len(example_info)} examples:",
                            className="mb-3",
                        ),
                        html.Table(
                            [
                                html.Thead(
                                    [
                                        html.Tr(
                                            [
                                                html.Th(
                                                    "Feature",
                                                    style={"width": "10%"},
                                                ),
                                                html.Th(
                                                    "Co-occurs",
                                                    style={"width": "10%"},
                                                ),
                                                html.Th(
                                                    "Rate",
                                                    style={"width": "10%"},
                                                ),
                                                html.Th(
                                                    "Avg Act.",
                                                    style={"width": "10%"},
                                                ),
                                                html.Th(
                                                    "Label",
                                                    style={"width": "30%"},
                                                ),
                                                html.Th(
                                                    "In Examples",
                                                    style={"width": "30%"},
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                html.Tbody(feature_rows),
                            ],
                            className="table table-sm table-hover",
                        ),
                        html.Hr(),
                        html.P(
                            [
                                html.Strong("Analysis Summary:"),
                                html.Br(),
                                f"• Most common co-activator: Feature {top_coactivating[0][0]} ({top_coactivating[0][1]}/{len(example_info)} examples)",
                                html.Br(),
                                f"• Average co-activating features per example: {sum(feature_counter.values()) / len(example_info):.1f}",
                            ],
                            className="small text-muted mt-3",
                        ),
                    ]
                ),
                example_features_map,
            )  # Return the map as second value

        else:
            # Legacy database - not implemented
            return (
                html.Div(
                    "Co-activation analysis not yet implemented for legacy database. Consider converting to efficient format.",
                    className="text-warning",
                ),
                {},
            )

    except Exception as e:
        logger.error(f"Error analyzing co-activations: {e}", exc_info=True)
        return (
            html.Div(
                f"Error analyzing co-activations: {str(e)}",
                className="text-danger",
            ),
            {},
        )


def update_example_cards_with_highlights(
    example_data, features_map, highlighted_feature_id
):
    """Update example cards with highlighting based on selected feature.

    Args:
        example_data: The stored example data from example-indices-store
        features_map: Map of feature_id -> list of example indices
        highlighted_feature_id: The feature ID to highlight examples for (None to clear)

    Returns:
        tuple: (updated cards, highlighted_feature_id for storage)
    """
    if not example_data or "full_data" not in example_data:
        return [], highlighted_feature_id

    full_data = example_data["full_data"]

    # Get list of example indices to highlight
    highlighted_indices = set()
    if (
        highlighted_feature_id is not None
        and features_map
        and str(highlighted_feature_id) in features_map
    ):
        highlighted_indices = set(features_map[str(highlighted_feature_id)])

    # Recreate cards with highlighting
    cards = []
    for rank, card_data in enumerate(full_data, 1):
        # Check if this example should be highlighted (rank-1 because enumerate starts at 1)
        is_highlighted = (rank - 1) in highlighted_indices

        cards.append(
            create_example_card(
                card_data,
                rank,
                card_id=f"example-card-{rank}",
                highlighted=is_highlighted,
            )
        )

    return cards, highlighted_feature_id


def navigate_to_feature(n_clicks_list):
    """Navigate to a different feature when clicked."""
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update

    # Extract feature ID from the clicked link
    prop_id = ctx.triggered[0]["prop_id"]
    feature_info = json.loads(prop_id.split(".")[0])

    return feature_info["feature"]
