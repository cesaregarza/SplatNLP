from typing import Optional

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html, no_update


cluster_map_component = html.Div(
    [
        html.H4("Feature Clusters", className="mb-3"),
        html.P(
            "Scatter of features in UMAP (if available) or PCA space, "
            "colored by cluster. The selected feature is highlighted. "
            "Labeled features show their labels on hover.",
            className="text-muted",
        ),
        dcc.Graph(id="cluster-scatter", style={"height": "700px"}),
        html.Div(id="cluster-selection-summary", className="mt-2"),
    ],
    className="mb-4",
)


def _cluster_dataframe():
    """Return the cluster dataframe from the global context."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    return getattr(DASHBOARD_CONTEXT, "feature_clusters", None)


def _feature_labels_manager():
    """Return the feature labels manager from the global context."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    return getattr(DASHBOARD_CONTEXT, "feature_labels_manager", None)


@callback(
    Output("cluster-scatter", "figure"),
    Output("cluster-selection-summary", "children"),
    [
        Input("feature-dropdown", "value"),
        Input("active-tab-store", "data"),
        Input("feature-labels-updated", "data"),  # Refresh when labels change
    ],
)
def update_cluster_scatter(
    selected_feature_id: Optional[int],
    active_tab: Optional[str],
    _labels_updated: Optional[int],
):
    # Lazy loading: skip if tab is not active
    if active_tab != "tab-clusters":
        return no_update, no_update

    df = _cluster_dataframe()
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No feature cluster data loaded",
            template="plotly_white",
            height=700,
        )
        return fig, dbc.Alert(
            "Load a feature_clusters parquet to enable this view.",
            color="warning",
            className="mt-2",
        )

    # Get feature labels manager
    labels_manager = _feature_labels_manager()

    x_col = "umap_x" if "umap_x" in df.columns else "pca_x"
    y_col = "umap_y" if "umap_y" in df.columns else "pca_y"

    hover_text = []
    top_tok_available = "top_tok" in df.columns
    for _, row in df.iterrows():
        feature_id = int(row["feature_id"])

        # Get user-provided label if available
        label_text = ""
        if labels_manager:
            label = labels_manager.get_label(feature_id)
            if label:
                label_text = f"<br><b>{label}</b>"

        top_tok = ""
        if top_tok_available and row["top_tok"]:
            top_tok = f"<br>top_tok: {row['top_tok']}"

        hover_text.append(
            f"feature {feature_id}{label_text}"
            f"<br>cluster {int(row['cluster'])}{top_tok}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                color=df["cluster"],
                colorscale="Viridis",
                size=5,
                opacity=0.6,
                showscale=False,
            ),
            hoverinfo="text",
            text=hover_text,
            name="features",
        )
    )

    summary_children = dbc.Alert(
        "Select a feature to see its cluster context.",
        color="info",
        className="mt-2",
    )

    if (
        selected_feature_id is not None
        and selected_feature_id in set(df["feature_id"].tolist())
    ):
        sel_row = df[df["feature_id"] == selected_feature_id].iloc[0]

        # Get label for selected feature
        selected_label = None
        if labels_manager:
            selected_label = labels_manager.get_label(selected_feature_id)

        # Build hover text for selected feature
        selected_hover = f"feature {selected_feature_id}"
        if selected_label:
            selected_hover += f"<br><b>{selected_label}</b>"
        selected_hover += f"<br>cluster {int(sel_row['cluster'])}"

        fig.add_trace(
            go.Scattergl(
                x=[sel_row[x_col]],
                y=[sel_row[y_col]],
                mode="markers",
                marker=dict(
                    color="red",
                    size=14,
                    line=dict(color="black", width=1),
                ),
                hoverinfo="text",
                text=[selected_hover],
                name="selected",
            )
        )
        cluster_id = int(sel_row["cluster"])
        cluster_size = int((df["cluster"] == cluster_id).sum())
        top_tok = (
            sel_row["top_tok"]
            if top_tok_available and sel_row.get("top_tok")
            else None
        )

        # Build summary with label
        summary_parts = [html.Span(f"Feature {selected_feature_id} ")]
        if selected_label:
            summary_parts.append(
                html.Span(
                    f'"{selected_label}" ',
                    className="fw-bold text-primary",
                )
            )
        summary_parts.extend(
            [
                html.Span(
                    f"in cluster {cluster_id} "
                    f"({cluster_size} features). "
                ),
                html.Span(
                    f"Top token: {top_tok}" if top_tok else "Top token: n/a"
                ),
            ]
        )

        summary_children = dbc.Alert(
            summary_parts,
            color="success",
            className="mt-2",
        )

    fig.update_layout(
        title="Feature clusters",
        template="plotly_white",
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig, summary_children
