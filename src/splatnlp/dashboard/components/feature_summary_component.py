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
    import logging

    import dash_bootstrap_components as dbc  # For layout

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    logger = logging.getLogger(__name__)

    if selected_feature_id is None:
        return [html.P("Select a feature to see its summary.")]

    if (
        DASHBOARD_CONTEXT is None
        or not hasattr(DASHBOARD_CONTEXT, "db")
        or DASHBOARD_CONTEXT.db is None
    ):
        logger.warning(
            "FeatureSummary: Dashboard context or DB context not available."
        )
        return [
            html.P(
                "Error: Database context not available.", style={"color": "red"}
            )
        ]

    db = DASHBOARD_CONTEXT.db
    stats = db.get_feature_stats(selected_feature_id)

    # Get feature display name
    # Assuming feature_labels_manager is the correct new name as per cli.py changes
    feature_labels_manager = getattr(
        DASHBOARD_CONTEXT, "feature_labels_manager", None
    )
    if feature_labels_manager:
        display_name = feature_labels_manager.get_display_name(
            selected_feature_id
        )
    else:
        display_name = f"Feature {selected_feature_id}"

    summary_elements = [
        html.H5(f"Summary for: {display_name}", className="mb-3")
    ]

    if not stats:
        summary_elements.append(
            html.P(f"No statistics found for feature {selected_feature_id}.")
        )
        return summary_elements

    def create_stat_row(label: str, value: Any, unit: str = ""):
        formatted_value = value
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif value is None:
            formatted_value = "N/A"

        return dbc.Row(
            [
                dbc.Col(
                    html.Strong(f"{label}:"), width="auto", className="pe-0"
                ),
                dbc.Col(f"{formatted_value} {unit}".strip()),
            ],
            className="mb-1",
        )

    summary_elements.extend(
        [
            create_stat_row("Min Activation", stats.get("min")),
            create_stat_row("Max Activation", stats.get("max")),
            create_stat_row("Mean", stats.get("mean")),
            create_stat_row("Median", stats.get("median")),
            create_stat_row("Standard Deviation", stats.get("std")),
            create_stat_row("25th Percentile", stats.get("q25")),
            create_stat_row("75th Percentile", stats.get("q75")),
            create_stat_row("Number of Zeros", stats.get("n_zeros")),
            create_stat_row("Total Examples", stats.get("n_total")),
            create_stat_row("Sparsity", f"{stats.get('sparsity', 0):.2%}"),
        ]
    )

    return summary_elements
