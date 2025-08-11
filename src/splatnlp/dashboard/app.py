import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import dash
import dash_bootstrap_components as dbc
from dash import (
    ALL,
    MATCH,
    Input,
    Output,
    State,
    callback_context,
    dcc,
    html,
    no_update,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.info("=== Starting app.py initialization ===")

# THIS IS WHERE THE GLOBAL CONTEXT WILL BE STORED
# It will be populated by the script that runs the dashboard (e.g., cli.py or run_dashboard.py)
DASHBOARD_CONTEXT = SimpleNamespace()  # Initialize as a SimpleNamespace
logger.info("DASHBOARD_CONTEXT created as SimpleNamespace")

from splatnlp.dashboard.efficient_fs_database import EfficientFSDatabase
from splatnlp.dashboard.fs_database import FSDatabase


def init_filesystem_database(
    meta_path: str = "/mnt/e/activations2/outputs/activations.metadata.joblib",
    neurons_root: str = "/mnt/e/activations2/outputs/neuron_acts",
) -> None:
    """Initialize the filesystem-based database connection.

    Args:
        meta_path: Path to the metadata joblib file
        neurons_root: Path to the root directory containing neuron_XXXX folders
    """
    DASHBOARD_CONTEXT.db = FSDatabase(meta_path, neurons_root)
    DASHBOARD_CONTEXT.feature_ids = DASHBOARD_CONTEXT.db.get_all_feature_ids()

    # Make the original analysis_df easily accessible for ancillary logic
    DASHBOARD_CONTEXT.analysis_df = DASHBOARD_CONTEXT.db.analysis_df
    DASHBOARD_CONTEXT.metadata = DASHBOARD_CONTEXT.db.metadata
    DASHBOARD_CONTEXT.pad_token_id = DASHBOARD_CONTEXT.vocab["<PAD>"]

    # Check if we have cached database with influence data
    if hasattr(DASHBOARD_CONTEXT.db, "influence_data"):
        DASHBOARD_CONTEXT.influence_data = DASHBOARD_CONTEXT.db.influence_data


def init_efficient_database(
    data_dir: str = "/mnt/e/activations_ultra_efficient",
    examples_dir: str = "/mnt/e/dashboard_examples_optimized",
) -> None:
    """Initialize the efficient database using optimized storage.

    Args:
        data_dir: Path to the Parquet/Zarr converted data
        examples_dir: Path to the optimized examples storage
    """
    DASHBOARD_CONTEXT.db = EfficientFSDatabase(data_dir, examples_dir)
    DASHBOARD_CONTEXT.feature_ids = DASHBOARD_CONTEXT.db.get_all_feature_ids()

    # For compatibility with existing code
    DASHBOARD_CONTEXT.analysis_df = None  # Not used in efficient version
    DASHBOARD_CONTEXT.metadata = DASHBOARD_CONTEXT.db.metadata
    if hasattr(DASHBOARD_CONTEXT, "vocab") and DASHBOARD_CONTEXT.vocab:
        DASHBOARD_CONTEXT.pad_token_id = DASHBOARD_CONTEXT.vocab.get(
            "<PAD>", 139
        )

    # Check if we have cached database with influence data
    if hasattr(DASHBOARD_CONTEXT.db, "influence_data"):
        DASHBOARD_CONTEXT.influence_data = DASHBOARD_CONTEXT.db.influence_data


# Create the app
logger.info("Creating Dash app...")
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
logger.info("Dash app created successfully")

# Import components AFTER app creation so callbacks can register properly
logger.info("Importing dashboard components...")
from splatnlp.dashboard.components import (
    ablation_component,
    activation_hist_component,
    correlations_component,
    example_features_component,
    feature_influence_component,
    feature_selector_layout,
    feature_summary_component,
    intervals_grid_component,
    token_analysis,
    top_examples_component,
    top_logits_component,
)
from splatnlp.dashboard.components.feature_labels import (
    FeatureLabelsManager,
    create_feature_label_editor,
    create_labeling_statistics,
)

# Placeholder for feature label editor - will be populated dynamically
feature_label_editor_container = html.Div(id="feature-label-editor-container")
labeling_statistics_container = html.Div(id="labeling-statistics-container")

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="page-load-trigger", storage_type="memory"),
        dcc.Store(id="feature-labels-updated", storage_type="memory", data=0),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("SAE Feature Dashboard", className="mb-4"),
                        feature_selector_layout,
                        feature_label_editor_container,  # Feature labeling editor
                        labeling_statistics_container,  # Labeling statistics
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Tabs(
                            id="analysis-tabs",
                            active_tab="tab-overview",
                            children=[
                                dbc.Tab(
                                    label="Overview",
                                    tab_id="tab-overview",
                                    children=[
                                        feature_summary_component,
                                        activation_hist_component,
                                    ],
                                ),
                                dbc.Tab(
                                    label="Top Examples",
                                    tab_id="tab-examples",
                                    children=top_examples_component,
                                ),
                                dbc.Tab(
                                    label="Intervals Grid",
                                    tab_id="tab-grid",
                                    children=intervals_grid_component,
                                ),
                                dbc.Tab(
                                    label="Co-Activation Analysis",
                                    tab_id="tab-example-features",
                                    children=example_features_component,
                                ),
                                dbc.Tab(
                                    label="Top Logits & Correlations",
                                    tab_id="tab-logits",
                                    children=[
                                        top_logits_component,
                                        correlations_component,
                                    ],
                                ),
                                dbc.Tab(
                                    label="Feature Influence",
                                    tab_id="tab-influence",
                                    children=feature_influence_component,
                                ),
                                dbc.Tab(
                                    label="Token Analysis",
                                    tab_id="tab-tokens",
                                    children=token_analysis.create_token_analysis_tab(),
                                ),
                                dbc.Tab(
                                    label="Ablation",
                                    tab_id="tab-ablation",
                                    children=ablation_component,
                                ),
                            ],
                        )
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# Register callbacks for the example features component
@app.callback(
    [
        Output("example-cards-container", "children"),
        Output("example-indices-store", "data"),
        Output("example-features-error", "children"),
    ],
    [Input("feature-dropdown", "value")],
)
def update_example_cards(selected_feature_id):
    """Load top examples for the selected feature and create cards."""
    logger.info(
        f"update_example_cards called with feature_id={selected_feature_id}"
    )
    logger.info(f"DASHBOARD_CONTEXT has db: {hasattr(DASHBOARD_CONTEXT, 'db')}")
    if hasattr(DASHBOARD_CONTEXT, "db"):
        logger.info(f"DASHBOARD_CONTEXT.db type: {type(DASHBOARD_CONTEXT.db)}")

    try:
        from splatnlp.dashboard.components.example_features_component import (
            update_example_cards as impl,
        )

        result = impl(selected_feature_id)
        logger.info(
            f"update_example_cards returning: {result[2] if len(result) > 2 else 'success'}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in update_example_cards: {e}", exc_info=True)
        return [], {}, f"Error: {str(e)}"


# New callback for co-activation analysis
@app.callback(
    [
        Output("coactivation-analysis-container", "children"),
        Output("example-features-map", "data"),
    ],
    [Input("analyze-coactivations-btn", "n_clicks")],
    [State("example-indices-store", "data")],
    prevent_initial_call=True,
)
def analyze_coactivations(n_clicks, example_indices_data):
    """Analyze co-activating features for all examples."""
    logger.info(f"analyze_coactivations called with n_clicks={n_clicks}")
    from splatnlp.dashboard.components.example_features_component import (
        analyze_coactivations as impl,
    )

    return impl(n_clicks, example_indices_data)


# DISABLED - Pattern matching callbacks cause issues with dashboard loading
# # Callback for highlighting examples when a feature is clicked
# @app.callback(
#     [
#         Output("example-cards-container", "children"),
#         Output("highlighted-feature", "data"),
#     ],
#     [Input({"type": "highlight-feature-btn", "index": ALL}, "n_clicks")],
#     [
#         State("example-indices-store", "data"),
#         State("example-features-map", "data"),
#         State("highlighted-feature", "data"),
#     ],
#     prevent_initial_call=True,
# )
# def highlight_examples_with_feature(n_clicks_list, example_data, features_map, current_highlighted):
#     """Highlight examples that have the selected co-activating feature."""
#     import dash
#     from splatnlp.dashboard.components.example_features_component import update_example_cards_with_highlights
#
#     ctx = dash.callback_context
#     if not ctx.triggered or not any(n_clicks_list):
#         return dash.no_update, dash.no_update
#
#     # Get the feature ID that was clicked
#     prop_id = ctx.triggered[0]["prop_id"]
#     feature_id = json.loads(prop_id.split(".")[0])["index"]
#
#     # Toggle highlight if clicking the same feature
#     if current_highlighted == feature_id:
#         feature_id = None
#
#     return update_example_cards_with_highlights(example_data, features_map, feature_id)

# DISABLED AGAIN - Pattern matching still causing issues
# @app.callback(
#     Output("feature-dropdown", "value"),
#     [Input({"type": "feature-link", "feature": ALL}, "n_clicks")],
#     prevent_initial_call=True,
# )
# def navigate_to_feature(n_clicks_list):
#     """Navigate to a different feature when clicked."""
#     try:
#         from splatnlp.dashboard.components.example_features_component import navigate_to_feature as impl
#         return impl(n_clicks_list)
#     except Exception as e:
#         logger.error(f"Error in navigate_to_feature: {e}")
#         return no_update


# This callback is to set the page-load-trigger once DASHBOARD_CONTEXT is potentially available.
# It's triggered by the dcc.Location component after the initial layout is rendered.
# The assumption is that the script running the Dash app (e.g., cli.py) will have set
# DASHBOARD_CONTEXT *before* app.run_server() is called.
@app.callback(Output("page-load-trigger", "data"), Input("url", "pathname"))
def trigger_page_load(pathname):
    """
    This callback fires when the page loads (or when the URL changes).
    It serves as a trigger for other callbacks that depend on DASHBOARD_CONTEXT.
    """
    import time

    logger.info(f"trigger_page_load called with pathname={pathname}")
    logger.info(
        f"DASHBOARD_CONTEXT attributes: {list(vars(DASHBOARD_CONTEXT).keys())}"
    )
    timestamp = time.time()
    logger.info(f"Returning timestamp: {timestamp}")
    return timestamp


# Feature Label Callbacks
@app.callback(
    Output("feature-label-editor-container", "children"),
    Input("feature-dropdown", "value"),
)
def update_feature_label_editor(selected_feature_id):
    """Update the feature label editor when a new feature is selected."""
    logger.info(
        f"update_feature_label_editor called with feature_id={selected_feature_id}"
    )
    logger.info(
        f"DASHBOARD_CONTEXT has feature_labels_manager: {hasattr(DASHBOARD_CONTEXT, 'feature_labels_manager')}"
    )

    if not hasattr(DASHBOARD_CONTEXT, "feature_labels_manager"):
        logger.warning("Feature labels manager not initialized")
        return html.Div(
            "Feature labels not initialized", className="text-muted"
        )

    if selected_feature_id is None:
        logger.info("No feature selected")
        return html.Div(
            "Select a feature to edit labels", className="text-muted"
        )

    logger.info(f"Creating label editor for feature {selected_feature_id}")
    return create_feature_label_editor(
        selected_feature_id, DASHBOARD_CONTEXT.feature_labels_manager
    )


@app.callback(
    [
        Output("feature-labels-feedback", "children"),
        Output("feature-labels-updated", "data"),
        Output("labeling-statistics-container", "children"),
    ],
    [
        Input("save-label-btn", "n_clicks"),
        Input("delete-label-btn", "n_clicks"),
    ],
    [
        State("feature-dropdown", "value"),
        State("feature-label-input", "value"),
        State("feature-confidence-dropdown", "value"),
        State("feature-labels-updated", "data"),
    ],
)
def handle_label_actions(
    save_clicks,
    delete_clicks,
    feature_id,
    label_text,
    confidence,
    update_counter,
):
    """Handle save and delete actions for feature labels."""
    from dash import callback_context

    if not callback_context.triggered:
        # Initial load - just show statistics
        if hasattr(DASHBOARD_CONTEXT, "feature_labels_manager"):
            stats = create_labeling_statistics(
                DASHBOARD_CONTEXT.feature_labels_manager
            )
            return "", update_counter, stats
        return "", update_counter, html.Div()

    if not hasattr(DASHBOARD_CONTEXT, "feature_labels_manager"):
        return (
            html.Div("Feature labels not initialized", className="text-danger"),
            update_counter,
            html.Div(),
        )

    button_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    manager = DASHBOARD_CONTEXT.feature_labels_manager

    if button_id == "save-label-btn" and feature_id is not None:
        if label_text:
            success = manager.set_label(feature_id, label_text, confidence)
            if success:
                # Update statistics
                stats = create_labeling_statistics(manager)
                return (
                    html.Div(
                        f"✓ Label saved for feature {feature_id}",
                        className="text-success",
                    ),
                    update_counter + 1,
                    stats,
                )
            return (
                html.Div("Failed to save label", className="text-danger"),
                update_counter,
                create_labeling_statistics(manager),
            )
        return (
            html.Div("Please enter a label", className="text-warning"),
            update_counter,
            create_labeling_statistics(manager),
        )

    elif button_id == "delete-label-btn" and feature_id is not None:
        success = manager.delete_label(feature_id)
        if success:
            stats = create_labeling_statistics(manager)
            return (
                html.Div(
                    f"✓ Label deleted for feature {feature_id}",
                    className="text-success",
                ),
                update_counter + 1,
                stats,
            )
        return (
            html.Div("No label to delete", className="text-info"),
            update_counter,
            create_labeling_statistics(manager),
        )

    return "", update_counter, create_labeling_statistics(manager)


# Add callbacks for new token analysis components
@app.callback(
    Output("single-token-examples", "children"),
    Input("feature-dropdown", "value"),
)
def update_single_token_examples(feature_id):
    """Update single token examples when feature selection changes."""
    if feature_id is None:
        return "Select a feature to see single token examples."

    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, "db"):
        return "Database context not available."

    examples = DASHBOARD_CONTEXT.db.get_single_token_examples(feature_id)
    if examples.empty:
        return "No single token examples found."

    return dbc.Table.from_dataframe(
        examples,
        striped=True,
        bordered=True,
        hover=True,
        size="sm",
        className="mt-3",
    )


@app.callback(
    Output("token-pair-examples", "children"),
    Input("feature-dropdown", "value"),
)
def update_token_pair_examples(feature_id):
    """Update token pair examples when feature selection changes."""
    if feature_id is None:
        return "Select a feature to see token pair examples."

    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, "db"):
        return "Database context not available."

    examples = DASHBOARD_CONTEXT.db.get_top_examples(feature_id)
    if examples.empty:
        return "No token pair examples found."

    return dbc.Table.from_dataframe(
        examples,
        striped=True,
        bordered=True,
        hover=True,
        size="sm",
        className="mt-3",
    )


@app.callback(
    Output("token-triple-examples", "children"),
    Input("feature-dropdown", "value"),
)
def update_token_triple_examples(feature_id):
    """Update token triple examples when feature selection changes."""
    if feature_id is None:
        return "Select a feature to see token triple examples."

    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, "db"):
        return "Database context not available."

    examples = DASHBOARD_CONTEXT.db.get_triple_examples(feature_id)
    if examples.empty:
        return "No token triple examples found."

    return dbc.Table.from_dataframe(
        examples,
        striped=True,
        bordered=True,
        hover=True,
        size="sm",
        className="mt-3",
    )


logger.info(f"=== app.py initialization complete ===")
logger.info(f"Total callbacks registered: {len(app.callback_map)}")
logger.info(f"App ready to run")
