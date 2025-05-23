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

# Feature names functionality is integrated directly

# THIS IS WHERE THE GLOBAL CONTEXT WILL BE STORED
# It will be populated by the script that runs the dashboard (e.g., cli.py or run_dashboard.py)
DASHBOARD_CONTEXT = SimpleNamespace()  # Initialize as a SimpleNamespace

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

# Create the feature name editor component
feature_name_editor = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Feature Name:"),
                dbc.Input(
                    id="feature-name-input",
                    placeholder="Enter a descriptive name for this feature",
                    type="text",
                    value="",
                ),
                dbc.Button(
                    "Save",
                    id="save-feature-name-btn",
                    color="primary",
                    n_clicks=0,
                ),
            ],
            className="mb-2",
        ),
        html.Div(id="feature-name-feedback", className="text-muted small"),
    ]
)

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
                        feature_name_editor,  # Add the editor directly here
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


# Feature name callbacks
@app.callback(
    Output("feature-name-input", "value"),
    Input("feature-dropdown", "value"),
    prevent_initial_call=True,
)
def load_feature_name(feature_id):
    """Load existing feature name when selection changes."""
    print(f"Load feature name callback: feature_id={feature_id}")
    if feature_id is None or feature_id == -1:
        return ""

    if (
        hasattr(DASHBOARD_CONTEXT, "feature_names_manager")
        and DASHBOARD_CONTEXT.feature_names_manager
    ):
        name = (
            DASHBOARD_CONTEXT.feature_names_manager.get_name(feature_id) or ""
        )
        print(f"Loaded name for feature {feature_id}: {name}")
        return name
    return ""


@app.callback(
    Output("feature-name-feedback", "children"),
    Output("feature-names-updated", "data"),
    Input("save-feature-name-btn", "n_clicks"),
    State("feature-dropdown", "value"),
    State("feature-name-input", "value"),
    State("feature-names-updated", "data"),
    prevent_initial_call=True,
)
def save_feature_name(n_clicks, feature_id, name, current_counter):
    """Save feature name when button is clicked."""
    print(
        f"Save callback triggered: n_clicks={n_clicks}, feature_id={feature_id}, name={name}"
    )

    # Always show something when clicked to verify callback is working
    if n_clicks is None:
        return "", current_counter or 0

    if n_clicks == 0:
        return "", current_counter or 0

    if feature_id is None or feature_id == -1:
        return "Please select a feature first.", current_counter or 0

    if (
        hasattr(DASHBOARD_CONTEXT, "feature_names_manager")
        and DASHBOARD_CONTEXT.feature_names_manager
    ):
        print(f"Setting name for feature {feature_id}: {name}")
        DASHBOARD_CONTEXT.feature_names_manager.set_name(feature_id, name)
        print(
            f"Current feature names: {DASHBOARD_CONTEXT.feature_names_manager.feature_names}"
        )
        print(
            f"Saved to file: {DASHBOARD_CONTEXT.feature_names_manager.storage_path}"
        )

        # Verify the name was saved
        saved_name = DASHBOARD_CONTEXT.feature_names_manager.get_name(
            feature_id
        )
        print(f"Verification - saved name: {saved_name}")

        # Increment counter to trigger dropdown refresh
        new_counter = (current_counter or 0) + 1

        if name.strip():
            return f"✓ Saved name for Feature {feature_id}", new_counter
        else:
            return f"✓ Removed name for Feature {feature_id}", new_counter

    return "Feature naming not available", current_counter or 0


if __name__ == "__main__":
    # This section is for development testing of the app structure.
    # The actual data loading and context setting will happen in the main script.
    print("Running in development mode. DASHBOARD_CONTEXT will be None.")
    print(
        "Functionality requiring DASHBOARD_CONTEXT (like feature list) might not work as expected."
    )
    app.run_server(debug=True)
