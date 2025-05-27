from typing import Any, List, Optional

from dash import Input, Output, callback, dcc, html

feature_summary_component = html.Div(
    id="feature-summary-content",
    children=[
        html.H4("Feature Summary", className="mb-3"),
        html.Div(id="selected-feature-display"),
        # Placeholder for auto-interpretation score
        # Placeholder for human explanation
        # Placeholder for ablation/prediction score
    ],
    className="mb-4",
)


@callback(
    Output("selected-feature-display", "children"),
    Input("feature-dropdown", "value"),
)
def update_feature_summary(selected_feature_id: Optional[int]) -> List[Any]:
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    import dash_bootstrap_components as dbc # For layout
    import logging
    logger = logging.getLogger(__name__)

    if selected_feature_id is None:
        return [html.P("Select a feature to see its summary.")]

    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, 'db_context') or DASHBOARD_CONTEXT.db_context is None:
        logger.warning("FeatureSummary: Dashboard context or DB context not available.")
        return [html.P("Error: Database context not available.", style={"color": "red"})]

    db_context = DASHBOARD_CONTEXT.db_context
    stats = db_context.get_feature_statistics(selected_feature_id)

    # Get feature display name
    # Assuming feature_labels_manager is the correct new name as per cli.py changes
    feature_labels_manager = getattr(DASHBOARD_CONTEXT, "feature_labels_manager", None)
    if feature_labels_manager:
        display_name = feature_labels_manager.get_display_name(selected_feature_id)
    else:
        display_name = f"Feature {selected_feature_id}"
    
    summary_elements = [
        html.H5(f"Summary for: {display_name}", className="mb-3")
    ]

    if not stats:
        summary_elements.append(html.P(f"No statistics found for feature {selected_feature_id}."))
        return summary_elements

    def create_stat_row(label: str, value: Any, unit: str = ""):
        formatted_value = value
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif value is None:
            formatted_value = "N/A"
        
        return dbc.Row([
            dbc.Col(html.Strong(f"{label}:"), width="auto", className="pe-0"),
            dbc.Col(f"{formatted_value} {unit}".strip())
        ], className="mb-1")

    summary_elements.extend([
        create_stat_row("Overall Min Activation", stats.get("overall_min_activation")),
        create_stat_row("Overall Max Activation", stats.get("overall_max_activation")),
        create_stat_row("Estimated Mean", stats.get("estimated_mean")),
        create_stat_row("Estimated Median", stats.get("estimated_median")),
        create_stat_row("Number of Samples in Bins", stats.get("num_sampled_examples")),
        create_stat_row("Number of Bins with Samples", stats.get("num_bins")),
    ])
    
    # Example of how to display histogram info (e.g., number of bins in stored histogram)
    # The actual histogram is displayed by another component.
    histogram_info = stats.get("histogram", {})
    if histogram_info and isinstance(histogram_info, dict):
        num_hist_bins = len(histogram_info.get("bin_ranges", []))
        summary_elements.append(create_stat_row("Number of Bins in Sampled Histogram", num_hist_bins))

    # Add placeholders for future items if desired
    # summary_elements.append(html.Hr(className="my-2"))
    # summary_elements.append(html.P("Auto-interpretation score: (coming soon)", className="text-muted"))
    # summary_elements.append(html.P("Human explanation: (coming soon)", className="text-muted"))

    return summary_elements
