from types import SimpleNamespace
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

# Import components
from splatnlp.dashboard.components import (
    activation_hist_component,
    correlations_component,
    feature_selector_layout,
    feature_summary_component,
    intervals_grid_component,
    top_examples_component,
    top_logits_component,
)
from splatnlp.dashboard.components.feature_labels import (
    create_feature_label_editor,
    create_labeling_statistics,
)

# THIS IS WHERE THE GLOBAL CONTEXT WILL BE STORED
# It will be populated by the script that runs the dashboard (e.g., cli.py or run_dashboard.py)
DASHBOARD_CONTEXT = SimpleNamespace()  # Initialize as a SimpleNamespace

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

# Placeholder for feature label editor - will be populated dynamically
feature_label_editor_container = html.Div(id="feature-label-editor-container")
labeling_statistics_container = html.Div(id="labeling-statistics-container")

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="page-load-trigger", storage_type="memory"),
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
                            [
                                dbc.Tab(
                                    label="Overview",
                                    children=[
                                        feature_summary_component,
                                        activation_hist_component,
                                    ],
                                ),
                                dbc.Tab(
                                    label="Top Examples",
                                    children=top_examples_component,
                                ),
                                dbc.Tab(
                                    label="Intervals Grid",
                                    children=intervals_grid_component,
                                ),
                                dbc.Tab(
                                    label="Top Logits & Correlations",
                                    children=[
                                        top_logits_component,
                                        correlations_component,
                                    ],
                                ),
                            ]
                        )
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# This callback is to set the page-load-trigger once DASHBOARD_CONTEXT is potentially available.
# It's triggered by the dcc.Location component after the initial layout is rendered.
# The assumption is that the script running the Dash app (e.g., cli.py) will have set
# DASHBOARD_CONTEXT *before* app.run_server() is called.
@app.callback(Output("page-load-trigger", "data"), Input("url", "pathname"))
def trigger_page_load(_pathname: Optional[str]) -> str:
    # This callback's main purpose is to signal that the initial app setup phase
    # (where context might be loaded externally by the script starting the server)
    # should be complete. The value returned (timestamp) is a simple way to trigger
    # other callbacks that depend on "page-load-trigger".
    import time

    return time.time()


# Feature label editor callbacks
@app.callback(
    Output("feature-label-editor-container", "children"),
    Input("feature-dropdown", "value"),
    prevent_initial_call=True,
)
def update_feature_label_editor(feature_id):
    """Update feature label editor when selection changes."""
    if feature_id is None or feature_id == -1:
        return html.Div("Select a feature to label", className="text-muted")

    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        return create_feature_label_editor(
            feature_id, DASHBOARD_CONTEXT.feature_labels_manager
        )
    return html.Div("Feature labeling not available", className="text-warning")


@app.callback(
    Output("feature-labels-feedback", "children"),
    Output("feature-labels-updated", "data"),
    Output("labeling-statistics-container", "children"),
    Input("save-feature-labels-btn", "n_clicks"),
    Input("clear-feature-labels-btn", "n_clicks"),
    State("feature-dropdown", "value"),
    State("feature-name-input", "value"),
    State("feature-category-radio", "value"),
    State("feature-notes-textarea", "value"),
    State("feature-labels-updated", "data"),
    prevent_initial_call=True,
)
def save_feature_labels(
    save_clicks,
    clear_clicks,
    feature_id,
    name,
    category,
    notes,
    current_counter,
):
    """Save or clear feature labels."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", current_counter or 0, dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if feature_id is None or feature_id == -1:
        return (
            "Please select a feature first.",
            current_counter or 0,
            dash.no_update,
        )

    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        if button_id == "save-feature-labels-btn":
            # Save labels
            DASHBOARD_CONTEXT.feature_labels_manager.update_label(
                feature_id,
                name=name or "",
                category=category or "none",
                notes=notes or "",
            )
            new_counter = (current_counter or 0) + 1
            stats_component = create_labeling_statistics(
                DASHBOARD_CONTEXT.feature_labels_manager
            )
            return (
                f"✓ Saved labels for Feature {feature_id}",
                new_counter,
                stats_component,
            )

        elif button_id == "clear-feature-labels-btn":
            # Clear labels
            DASHBOARD_CONTEXT.feature_labels_manager.update_label(
                feature_id, name="", category="none", notes=""
            )
            new_counter = (current_counter or 0) + 1
            stats_component = create_labeling_statistics(
                DASHBOARD_CONTEXT.feature_labels_manager
            )
            return (
                f"✓ Cleared labels for Feature {feature_id}",
                new_counter,
                stats_component,
            )

    return (
        "Feature labeling not available",
        current_counter or 0,
        dash.no_update,
    )


# Update statistics on page load
@app.callback(
    Output("labeling-statistics-container", "children", allow_duplicate=True),
    Input("page-load-trigger", "data"),
    prevent_initial_call=True,
)
def update_statistics_on_load(_):
    """Update labeling statistics on page load."""
    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        return create_labeling_statistics(
            DASHBOARD_CONTEXT.feature_labels_manager
        )
    return html.Div()


if __name__ == "__main__":
    # This section is for development testing of the app structure.
    # The actual data loading and context setting will happen in the main script.
    print("Running in development mode. DASHBOARD_CONTEXT will be None.")
    print(
        "Functionality requiring DASHBOARD_CONTEXT (like feature list) might not work as expected."
    )
    app.run_server(debug=True)
