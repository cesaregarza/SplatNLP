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
    if selected_feature_id is None:
        return [html.P("Select a feature to see its summary.")]

    return [
        html.P(f"Selected Feature: {selected_feature_id}"),
    ]


# Note:
# For now, we are only displaying the feature ID.
# We will need to figure out how to get/calculate:
# - Auto-interpretation score
# - Human explanation (this might be a manual annotation process outside the app's direct scope for now)
# - Ablation/prediction score
# These will likely require access to the DASHBOARD_CONTEXT or other data sources.
