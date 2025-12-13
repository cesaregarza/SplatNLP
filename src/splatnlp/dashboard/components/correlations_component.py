from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from dash import Input, Output, State, callback, dcc, html, no_update

# App context will be monkey-patched by the run script
# DASHBOARD_CONTEXT = None

correlations_component = html.Div(
    id="correlations-content",
    children=[
        html.H4("Feature Correlations", className="mb-3"),
        dcc.Loading(
            id="loading-correlations",
            type="default",
            children=[
                html.Div(
                    id="sae-feature-correlations-display", className="mb-3"
                ),
                html.Hr(),
                html.Div(id="token-logit-correlations-display"),
            ],
            className="mb-2",
        ),
        html.P(id="correlations-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


@callback(
    [
        Output("sae-feature-correlations-display", "children"),
        Output("token-logit-correlations-display", "children"),
        Output("correlations-error-message", "children"),
    ],
    [
        Input("feature-dropdown", "value"),
        Input("active-tab-store", "data"),
    ],
)
def update_correlations_display(selected_feature_id, active_tab):
    import logging

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    logger = logging.getLogger(__name__)

    # Lazy loading: skip if tab is not active
    if active_tab != "tab-logits":
        return no_update, no_update, no_update

    if selected_feature_id is None:
        return [], [], "Select an SAE feature to view correlations."

    if (
        DASHBOARD_CONTEXT is None
        or not hasattr(DASHBOARD_CONTEXT, "db")
        or DASHBOARD_CONTEXT.db is None
    ):
        logger.warning(
            "CorrelationsComponent: Dashboard context or DB context not available."
        )
        return (
            [],
            [],
            "Error: Database context not available. Ensure data is loaded correctly.",
        )

    db = DASHBOARD_CONTEXT.db
    error_message_parts = []
    sae_corr_display_children = []
    token_logit_influences_display_children = []

    try:
        # Get activations for the selected feature
        selected_activations = db.get_feature_activations(selected_feature_id)

        # Check if it's Polars or Pandas and handle empty case
        if isinstance(selected_activations, pl.DataFrame):
            if selected_activations.is_empty():
                return (
                    [],
                    [],
                    f"No activation data found for feature {selected_feature_id}",
                )
        else:  # pandas
            if selected_activations.empty:
                return (
                    [],
                    [],
                    f"No activation data found for feature {selected_feature_id}",
                )

        # For sparse features, correlation is often not meaningful
        # since features rarely activate on the same examples
        # Skip correlation calculation for efficient database
        correlations = []

        # Only compute correlations for small feature sets or non-efficient databases
        if db.__class__.__name__ != "EfficientFSDatabase":
            # Original correlation logic for FSDatabase
            for other_feature_id in DASHBOARD_CONTEXT.feature_ids:
                if other_feature_id == selected_feature_id:
                    continue

                other_activations = db.get_feature_activations(other_feature_id)

                # Check for empty
                if isinstance(other_activations, pl.DataFrame):
                    if other_activations.is_empty():
                        continue
                else:
                    if other_activations.empty:
                        continue

                # Calculate correlation between activations
                if isinstance(selected_activations, pl.DataFrame):
                    # Polars correlation - need to join on common examples
                    # Join on index to get common examples
                    joined = selected_activations.select(
                        ["index", "activation"]
                    ).join(
                        other_activations.select(["index", "activation"]),
                        on="index",
                        how="inner",
                        suffix="_other",
                    )

                    if (
                        len(joined) < 3
                    ):  # Need at least 3 points for meaningful correlation
                        continue

                    # Calculate correlation on common examples
                    selected_acts = joined["activation"].to_numpy()
                    other_acts = joined["activation_other"].to_numpy()

                    # Check for zero variance
                    if np.std(selected_acts) == 0 or np.std(other_acts) == 0:
                        continue

                    correlation = np.corrcoef(selected_acts, other_acts)[0, 1]
                else:
                    # Pandas correlation - join on index
                    joined = pd.merge(
                        selected_activations[["index", "activation"]],
                        other_activations[["index", "activation"]],
                        on="index",
                        how="inner",
                        suffixes=("", "_other"),
                    )

                    if len(joined) < 3:
                        continue

                    correlation = joined["activation"].corr(
                        joined["activation_other"]
                    )

                correlations.append(
                    {"feature_id": other_feature_id, "correlation": correlation}
                )

        # Sort by absolute correlation and take top 5
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_correlations = correlations[:5]

        sae_corr_display_children.append(
            html.H5("Top Correlated SAE Features", className="mb-2")
        )
        if top_correlations:
            for item in top_correlations:
                sae_corr_display_children.append(
                    html.P(
                        f"Feature {item['feature_id']}: {item['correlation']:.3f}",
                        className="ms-3",
                    )
                )
        else:
            sae_corr_display_children.append(
                html.P("No SAE feature correlations found.")
            )

        # Get ability tag correlations
        token_logit_influences_display_children.append(
            html.H5("Ability Tag Correlations", className="mb-2")
        )

        # Since ability_tags is an array, we need to expand it to calculate correlations
        try:
            if not isinstance(selected_activations, pd.DataFrame) or (
                "ability_tags" not in selected_activations.columns
            ):
                token_logit_influences_display_children.append(
                    html.P(
                        "Ability tag correlations unavailable for this backend.",
                        className="ms-3",
                    )
                )
                return (
                    sae_corr_display_children,
                    token_logit_influences_display_children,
                    "",
                )

            # Create a dictionary to store mean activations per ability tag
            tag_activations = {}

            for _, row in selected_activations.iterrows():
                activation = row["activation"]
                ability_tags = row["ability_tags"]

                # Handle numpy array or list format
                if isinstance(ability_tags, np.ndarray):
                    tags = ability_tags.tolist()
                elif isinstance(ability_tags, list):
                    tags = ability_tags
                else:
                    # Skip if format is unexpected
                    continue

                # Accumulate activations for each tag
                for tag in tags:
                    if tag not in tag_activations:
                        tag_activations[tag] = []
                    tag_activations[tag].append(activation)

            # Calculate mean activation per tag
            ability_tag_means = {}
            for tag, acts in tag_activations.items():
                ability_tag_means[tag] = np.mean(acts)

            # Sort by mean activation
            sorted_tags = sorted(
                ability_tag_means.items(), key=lambda x: x[1], reverse=True
            )

            if sorted_tags:
                # Top positive correlations
                token_logit_influences_display_children.append(
                    html.Strong("Top Positive Correlations:", className="ms-3")
                )
                for tag, mean_act in sorted_tags[:5]:
                    tag_name = DASHBOARD_CONTEXT.inv_vocab.get(
                        str(tag), f"Token_{tag}"
                    )
                    token_logit_influences_display_children.append(
                        html.P(
                            f"Ability '{tag_name}': Mean Activation {mean_act:.3f}",
                            className="ms-4",
                        )
                    )

                # Bottom correlations (lowest activations)
                token_logit_influences_display_children.append(
                    html.Strong("Lowest Correlations:", className="ms-3 mt-2")
                )
                for tag, mean_act in sorted_tags[-5:]:
                    tag_name = DASHBOARD_CONTEXT.inv_vocab.get(
                        str(tag), f"Token_{tag}"
                    )
                    token_logit_influences_display_children.append(
                        html.P(
                            f"Ability '{tag_name}': Mean Activation {mean_act:.3f}",
                            className="ms-4",
                        )
                    )
            else:
                token_logit_influences_display_children.append(
                    html.P("No ability tag correlations found.")
                )

        except Exception as e:
            logger.error(
                f"Error calculating ability tag correlations: {e}",
                exc_info=True,
            )
            token_logit_influences_display_children.append(
                html.P(f"Error calculating correlations: {str(e)}")
            )

        return (
            sae_corr_display_children,
            token_logit_influences_display_children,
            "",
        )

    except Exception as e:
        logger.error(f"Error in correlations display: {e}", exc_info=True)
        return [], [], f"Error calculating correlations: {str(e)}"
