from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from dash import Input, Output, State, callback, dcc, html

# App context will be monkey-patched by the run script
# DASHBOARD_CONTEXT = None # This will be set by run_dashboard.py or cli.py

top_logits_component = html.Div(
    id="top-logits-content",
    children=[
        html.H4(
            "Top Output Logits Influenced by SAE Feature", className="mb-3"
        ),
        dcc.Loading(
            id="loading-top-logits",
            type="default",
            children=dcc.Graph(id="top-logits-graph"),
            className="mb-2",
        ),
        html.P(id="top-logits-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


@callback(
    [
        Output("top-logits-graph", "figure"),
        Output("top-logits-error-message", "children"),
    ],
    [Input("feature-dropdown", "value")],
)
def update_top_logits_graph(
    selected_feature_id: Optional[int],
) -> Tuple[Dict[str, Any], str]:
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return {
            "data": [],
            "layout": {"title": "Select a feature to see its logit influences"},
        }, ""

    if DASHBOARD_CONTEXT is None:
        return {
            "data": [],
            "layout": {"title": "Dashboard context not initialized"},
        }, "Error: Dashboard context not initialized"

    sae_model = DASHBOARD_CONTEXT.sae_model
    if sae_model is None:
        return {
            "data": [],
            "layout": {"title": "SAE model not loaded"},
        }, "Error: SAE model not loaded"

    if not hasattr(sae_model, "decoder") or not hasattr(
        sae_model.decoder, "weight"
    ):
        return {
            "data": [],
            "layout": {"title": "SAE model missing decoder weights"},
        }, "Error: SAE model missing decoder weights"

    # Removed direct model access and on-the-fly calculation.
    # Now, we fetch precomputed logit influences from the database.
    # DASHBOARD_CONTEXT.inv_vocab is not needed here if token names are in the DB results.
    import logging
    logger = logging.getLogger(__name__)

    db_context = getattr(DASHBOARD_CONTEXT, 'db_context', None)
    if db_context is None:
        logger.warning("TopLogits: db_context not available.")
        return {"data": [], "layout": {"title": "Database context not available"}}, "Error: Database context not available"

    # Define how many top positive/negative influences to show.
    # This limit is passed to get_logit_influences.
    # The old code hardcoded 5 positive and 5 negative.
    limit_per_type = 5 

    try:
        logit_influences_data = db_context.get_logit_influences(selected_feature_id, limit=limit_per_type)
    except Exception as e:
        logger.error(f"TopLogits: Error fetching logit influences for feature {selected_feature_id}: {e}", exc_info=True)
        return {"data": [], "layout": {"title": f"Error fetching data for feature {selected_feature_id}"}}, f"Error: {str(e)}"

    positive_influences = logit_influences_data.get("positive", [])
    negative_influences = logit_influences_data.get("negative", [])

    if not positive_influences and not negative_influences:
        return {
            "data": [],
            "layout": {"title": f"No logit influence data for Feature {selected_feature_id}"},
        }, ""

    # Prepare data for DataFrame
    tokens = []
    effects = []
    types = []

    for item in positive_influences:
        tokens.append(item.get("token_name", f"ID_{item.get('token_id')}"))
        effects.append(item.get("influence_value")) # Use 'influence_value' as per DB schema
        types.append("Positive")
    
    for item in negative_influences:
        tokens.append(item.get("token_name", f"ID_{item.get('token_id')}"))
        effects.append(item.get("influence_value")) # Use 'influence_value'
        types.append("Negative")
        
    # Ensure all lists have compatible data for DataFrame creation
    # Filter out any entries where effect might be None if that's possible from DB
    valid_indices = [i for i, effect in enumerate(effects) if effect is not None]
    tokens = [tokens[i] for i in valid_indices]
    effects = [effects[i] for i in valid_indices]
    types = [types[i] for i in valid_indices]


    if not tokens: # If all influences were filtered out (e.g. all had None effect)
         return {
            "data": [],
            "layout": {"title": f"No valid logit influence data to display for Feature {selected_feature_id}"},
        }, ""


    df = pd.DataFrame({"Token": tokens, "Effect": effects, "Type": types})
    
    # Sort by Type (Positive first) then by absolute effect for consistent plotting if needed,
    # though rank from DB should already ensure order within type.
    # For display, usually positive influences are on one side, negative on other, or intermingled.
    # Plotly Express bar chart will group by color based on 'Type'.
    # If we want specific order of bars (e.g. highest positive to lowest positive, then lowest negative to highest negative),
    # we might need to sort df before passing to px.bar
    df = df.sort_values(by=['Type', 'Effect'], ascending=[True, False]) # Positive descending, Negative ascending (more negative)

    fig = px.bar(
        df,
        x="Token",
        y="Effect",
        color="Type",
        title=f"Top Logit Influences for Feature {selected_feature_id}",
        labels={"Token": "Token", "Effect": "Logit Influence"},
        color_discrete_map={"Positive": "#1f77b4", "Negative": "#ff7f0e"},
        category_orders={"Token": df["Token"].tolist()} # Preserve sorted order in plot
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
        xaxis={"tickangle": -45},  # Rotate labels for better readability
    )

    return fig, ""
