from typing import Any, Dict, Optional

import numpy as np
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html

# App context will be monkey-patched by the run script
# We need to declare it here for the callback to access it.
# This is a common pattern in multi-file Dash apps.
# See: https://dash.plotly.com/sharing-data-between-callbacks
# DASHBOARD_CONTEXT = None # This will be set by run_dashboard.py or cli.py

activation_hist_component = html.Div(
    [
        html.H4("Feature Activation Histogram", className="mb-3"),
        dcc.RadioItems(
            id="activation-filter-radio",
            options=[
                {"label": "All Activations", "value": "all"},
                {"label": "Non-Zero Activations", "value": "nonzero"},
            ],
            value="nonzero",
            labelStyle={"display": "inline-block", "margin-right": "10px"},
            className="mb-2",
        ),
        dcc.Graph(id="activation-histogram"),
    ],
    className="mb-4",
)


@callback(
    Output("activation-histogram", "figure"),
    [
        Input("feature-dropdown", "value"),
        Input("activation-filter-radio", "value"),
    ],
    # We need to access the global app context for the data
    # This relies on DASHBOARD_CONTEXT being set in the app's scope
    # For multi-page apps or more complex scenarios, Dash Enterprise's Job Queue
    # or other caching mechanisms (Redis, etc.) might be better.
    # For this project, accessing a module-level global is acceptable given the run script's setup.
    State("feature-dropdown", "value"),
)
def update_activation_histogram(
    selected_feature_id: Optional[int],
    filter_type: str,
    _: Optional[int],
) -> Dict[str, Any]:
    # This assumes that 'splatnlp.dashboard.app' module has DASHBOARD_CONTEXT attribute set by the script.
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None or DASHBOARD_CONTEXT is None:
        return {
            "data": [],
            "layout": {
                "title": "Select a feature to see its activation histogram"
            },
        }

    all_activations = DASHBOARD_CONTEXT.all_sae_hidden_activations
    # all_sae_hidden_activations is expected to be a NumPy array of shape (num_examples, num_features)

    if selected_feature_id >= all_activations.shape[1]:
        return {
            "data": [],
            "layout": {
                "title": f"Feature ID {selected_feature_id} out of range."
            },
        }

    feature_activations = all_activations[:, selected_feature_id]

    if filter_type == "nonzero":
        plot_activations = feature_activations[
            feature_activations > 1e-6
        ]  # Using a small epsilon for float comparison
        title = f"Non-Zero Activations for Feature {selected_feature_id}"
        if plot_activations.size == 0:
            plot_activations = np.array(
                [0.0]
            )  # Ensure hist has some data to prevent error
            title = f"No Non-Zero Activations for Feature {selected_feature_id}"
    else:
        plot_activations = feature_activations
        title = f"All Activations for Feature {selected_feature_id}"

    if (
        plot_activations.ndim == 0
    ):  # Handle case where there's only one value (e.g. a single zero)
        plot_activations = np.array([plot_activations.item()])

    fig = px.histogram(
        x=plot_activations,
        title=title,
        labels={"x": "Activation Value", "y": "Count"},
        nbins=50,
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,  # Set fixed height for the histogram
    )

    return fig
