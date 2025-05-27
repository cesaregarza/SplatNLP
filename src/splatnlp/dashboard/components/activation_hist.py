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
    State("feature-dropdown", "value"),
)
def update_activation_histogram(
    selected_feature_id: Optional[int],
    filter_type: str,
    _: Optional[int],
) -> Dict[str, Any]:
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None or DASHBOARD_CONTEXT is None:
        return {
            "data": [],
            "layout": {
                "title": "Select a feature to see its activation histogram"
            },
        }

    # Use database-backed context if available
    if hasattr(DASHBOARD_CONTEXT, 'db_context'):
        # Get precomputed histogram data from database
        db_context = DASHBOARD_CONTEXT.db_context
        stats = db_context.get_feature_statistics(selected_feature_id)
        
        if not stats or 'histogram' not in stats:
            return {
                "data": [],
                "layout": {
                    "title": f"No histogram data for Feature {selected_feature_id}"
                },
            }
        
        histogram_data = stats['histogram']
        counts = np.array(histogram_data['counts'])
        bin_edges = np.array(histogram_data['bin_edges'])
        
        # Filter based on user selection
        if filter_type == "nonzero":
            # Only show bins with activation > 1e-6
            nonzero_mask = bin_edges[:-1] > 1e-6
            counts = counts[nonzero_mask]
            bin_edges = bin_edges[np.concatenate([nonzero_mask, [False]])]
            title = f"Non-Zero Activations for Feature {selected_feature_id}"
            
            if len(counts) == 0:
                return {
                    "data": [],
                    "layout": {
                        "title": f"No non-zero activations for Feature {selected_feature_id}"
                    },
                }
        else:
            title = f"All Activations for Feature {selected_feature_id}"
        
        # Create histogram figure from precomputed data
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig = {
            "data": [{
                "type": "bar",
                "x": bin_centers,
                "y": counts,
                "name": "Count"
            }],
            "layout": {
                "title": title,
                "xaxis": {"title": "Activation Value"},
                "yaxis": {"title": "Count"},
                "showlegend": False,
                "margin": dict(l=40, r=40, t=40, b=40),
                "height": 300,
            }
        }
        
        return fig
    
    else:
        # Fallback to old method if database context not available
        all_activations = DASHBOARD_CONTEXT.all_sae_hidden_activations

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
            ]
            title = f"Non-Zero Activations for Feature {selected_feature_id}"
            if plot_activations.size == 0:
                plot_activations = np.array([0.0])
                title = f"No Non-Zero Activations for Feature {selected_feature_id}"
        else:
            plot_activations = feature_activations
            title = f"All Activations for Feature {selected_feature_id}"

        if plot_activations.ndim == 0:
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
            height=300,
        )

        return fig
