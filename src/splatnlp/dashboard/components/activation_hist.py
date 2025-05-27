from typing import Any, Dict, List, Optional

import plotly.graph_objects as go # Changed from plotly.express
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
State("feature-dropdown", "value"), # Keep state for consistency, though not strictly needed if using selected_feature_id
)
def update_activation_histogram(
    selected_feature_id: Optional[int],
    filter_type: str,
    _: Optional[int], # State variable, not used directly
) -> go.Figure: # Return type is now go.Figure
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    import logging
    logger = logging.getLogger(__name__)

    fig = go.Figure()
    fig.update_layout(
        title="Select a feature to see its activation histogram",
        xaxis_title="Activation Value",
        yaxis_title="Count of Sampled Examples",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
    )

    if selected_feature_id is None:
        return fig # Return empty fig with default title

    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, 'db_context') or DASHBOARD_CONTEXT.db_context is None:
        logger.warning("ActivationHist: Dashboard context or DB context not available.")
        fig.update_layout(title="Error: Database context not available.")
        return fig

    db_context = DASHBOARD_CONTEXT.db_context
    stats = db_context.get_feature_statistics(selected_feature_id)
    
    if not stats or not stats.get('histogram'):
        fig.update_layout(title=f"No histogram data for Feature {selected_feature_id}")
        return fig
    
    histogram_data = stats['histogram']
    # New structure: histogram_data = {"bin_ranges": [[min,max], ...], "counts": [c1, ...]}
    bin_ranges: List[List[float]] = histogram_data.get("bin_ranges", [])
    counts: List[int] = histogram_data.get("counts", [])

    if not bin_ranges or not counts or len(bin_ranges) != len(counts):
        logger.warning(f"Malformed histogram data for feature {selected_feature_id}: ranges={bin_ranges}, counts={counts}")
        fig.update_layout(title=f"Malformed histogram data for Feature {selected_feature_id}")
        return fig

    x_midpoints: List[float] = []
    y_counts: List[int] = []
    bar_widths: List[float] = []
    hover_texts: List[str] = []

    base_title = f"Histogram of Sampled Activations for Feature {selected_feature_id}"
    filter_desc = "(All Bins)"

    for i, (range_pair, count) in enumerate(zip(bin_ranges, counts)):
        if len(range_pair) != 2:
            logger.warning(f"Skipping malformed bin_range {range_pair} for feature {selected_feature_id}")
            continue
        
        min_act, max_act = range_pair[0], range_pair[1]

        if filter_type == "nonzero":
            filter_desc = "(Non-Zero Activation Bins)"
            # A bin is considered "non-zero" if its lower bound is > a small epsilon,
            # or if its upper bound is <= a small negative epsilon (to exclude bins around zero).
            # More simply, if the range itself does not strictly contain zero.
            # For positive activations, this usually means min_act > 1e-6.
            # If a bin is e.g. [0.0, 0.1), it includes values very close to zero.
            # If a bin is [-0.1, 0.0), it also includes values very close to zero.
            # A simple check: if min_act >= 0 and max_act <= 1e-6 (for a bin like [0, 0]), it's effectively zero.
            # Or, if the *majority* of the bin is non-zero.
            # Let's define "non-zero bin" as one where min_act > 1e-6
            if not (min_act > 1e-6): # If min_act is not strictly positive, skip this bin
                continue
        
        x_midpoints.append((min_act + max_act) / 2)
        y_counts.append(count)
        bar_widths.append(max_act - min_act)
        hover_texts.append(f"Range: [{min_act:.3g} - {max_act:.3g})<br>Count: {count}")

    if not x_midpoints: # If filter removed all bins
        if filter_type == "nonzero":
             fig.update_layout(title=f"No strictly positive activation bins for Feature {selected_feature_id}")
        else: # Should not happen if original data was not empty
             fig.update_layout(title=f"No data to display for Feature {selected_feature_id}")
        return fig

    fig.add_trace(go.Bar(
        x=x_midpoints,
        y=y_counts,
        width=bar_widths,
        hovertext=hover_texts,
        hoverinfo="text",
        name="Count"
    ))
    
    fig.update_layout(title=f"{base_title} {filter_desc}")
    return fig
