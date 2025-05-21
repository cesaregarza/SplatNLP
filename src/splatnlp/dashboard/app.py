import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# Import components
from splatnlp.dashboard.components import (
    feature_selector_layout,
    feature_summary_component,
    activation_hist_component,
    top_logits_component,
    top_examples_component,
    intervals_grid_component,
    correlations_component,
)

from types import SimpleNamespace

# THIS IS WHERE THE GLOBAL CONTEXT WILL BE STORED
# It will be populated by the script that runs the dashboard (e.g., cli.py or run_dashboard.py)
DASHBOARD_CONTEXT = SimpleNamespace() # Initialize as a SimpleNamespace

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

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
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        feature_summary_component,
                        activation_hist_component,
                        top_logits_component,
                        top_examples_component,
                        intervals_grid_component,
                        correlations_component,
                    ],
                    width=9,
                ),
            ]
        )
    ],
    fluid=True,
)

# This callback is to set the page-load-trigger once DASHBOARD_CONTEXT is potentially available.
# It's triggered by the dcc.Location component after the initial layout is rendered.
# The assumption is that the script running the Dash app (e.g., cli.py) will have set
# DASHBOARD_CONTEXT *before* app.run_server() is called.
@app.callback(
    Output("page-load-trigger", "data"),
    Input("url", "pathname")
)
def trigger_page_load(_pathname):
    # This callback's main purpose is to signal that the initial app setup phase
    # (where context might be loaded externally by the script starting the server)
    # should be complete. The value returned (timestamp) is a simple way to trigger
    # other callbacks that depend on "page-load-trigger".
    import time
    return time.time()


if __name__ == "__main__":
    # This section is for development testing of the app structure.
    # The actual data loading and context setting will happen in the main script.
    print("Running in development mode. DASHBOARD_CONTEXT will be None.")
    print("Functionality requiring DASHBOARD_CONTEXT (like feature list) might not work as expected.")
    app.run_server(debug=True)
