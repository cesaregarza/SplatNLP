from typing import List, Optional, Tuple

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

# DASHBOARD_CONTEXT will be populated by the main script in app.py

feature_selector_layout = html.Div(
    [
        html.Label("Select SAE Feature:", className="form-label mb-2"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[],  # Options will be populated by callback
            value=None,  # Default selection, will be updated by URL or set to 0
            clearable=False,
            searchable=True,
        ),
    ],
    className="mb-4",
)


@callback(
    Output("feature-dropdown", "options"),
    Output("feature-dropdown", "value"),
    Input("page-load-trigger", "data"),
    Input("feature-labels-updated", "data"),
    State("feature-dropdown", "value"),
    State("url", "search"),
)
def populate_feature_options(
    page_load_data: Optional[str],
    labels_updated_counter: Optional[int],
    current_value: Optional[int],
    search_query: Optional[str],
) -> Tuple[List[dict], Optional[int]]:
    import logging

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    logger = logging.getLogger(__name__)

    logger.debug(
        f"Dropdown refresh triggered, labels_updated_counter: {labels_updated_counter}"
    )

    options: List[dict] = []
    default_value: Optional[int] = (
        None  # Will be set to the first available feature or None
    )

    if (
        DASHBOARD_CONTEXT is None
        or not hasattr(DASHBOARD_CONTEXT, "db")
        or DASHBOARD_CONTEXT.db is None
    ):
        logger.warning(
            "FeatureSelector: Dashboard context or DB context not available."
        )
        options = [
            {
                "label": "Error: DB context not loaded",
                "value": -1,
                "disabled": True,
            }
        ]
        default_value = -1
    else:
        db = DASHBOARD_CONTEXT.db
        try:
            feature_ids = (
                db.get_all_feature_ids()
            )  # Fetch sorted list of feature IDs
            if feature_ids:
                feature_labels_manager = getattr(
                    DASHBOARD_CONTEXT, "feature_labels_manager", None
                )
                for feature_id in feature_ids:
                    if feature_labels_manager:
                        label = feature_labels_manager.get_display_name(
                            feature_id
                        )
                    else:
                        label = f"Feature {feature_id}"
                    options.append({"label": label, "value": feature_id})

                if options:  # If any valid features were found
                    default_value = options[0][
                        "value"
                    ]  # Default to the first feature in the sorted list
                else:  # Should not happen if feature_ids is not empty, but as a safeguard
                    options = [
                        {
                            "label": "No features found in DB",
                            "value": -1,
                            "disabled": True,
                        }
                    ]
                    default_value = -1

            else:  # No feature IDs returned from DB
                options = [
                    {
                        "label": "No features available in DB",
                        "value": -1,
                        "disabled": True,
                    }
                ]
                default_value = -1
        except Exception as e:
            logger.error(
                f"FeatureSelector: Error fetching feature IDs from DB: {e}",
                exc_info=True,
            )
            options = [
                {
                    "label": "Error loading features",
                    "value": -1,
                    "disabled": True,
                }
            ]
            default_value = -1

    # Determine initial value: URL query param > current_value (if valid) > default_value from DB list
    final_value = default_value

    if search_query:
        try:
            query_params = dict(
                qc.split("=")
                for qc in search_query.lstrip("?").split("&")
                if "=" in qc
            )
            feature_val_str = query_params.get("feature")
            if feature_val_str is not None:
                feature_val_from_url = int(feature_val_str)
                if any(
                    opt["value"] == feature_val_from_url
                    for opt in options
                    if not opt.get("disabled")
                ):
                    final_value = feature_val_from_url
        except Exception as e:
            logger.warning(
                f"FeatureSelector: Error parsing URL query for feature: {e}"
            )
            # Keep final_value as default_value if URL parsing fails or feature is invalid

    # If URL did not set a valid feature, and a current_value exists and is valid, keep it.
    # This handles cases where user has selected something, then page reloads for other reasons (e.g. label update).
    elif current_value is not None and any(
        opt["value"] == current_value
        for opt in options
        if not opt.get("disabled")
    ):
        final_value = current_value

    # Ensure final_value is sensible if the list of options is empty or only contains disabled items.
    if not options or all(opt.get("disabled") for opt in options):
        final_value = -1  # Or some other indicator of no valid selection
        if not options:  # If options list itself is empty, add a placeholder
            options = [
                {
                    "label": "No selectable features",
                    "value": -1,
                    "disabled": True,
                }
            ]

    logger.debug(
        f"Populating feature dropdown. Options count: {len(options)}. Final value: {final_value}"
    )
    return options, final_value


@callback(
    Output("url", "search"),
    Input("feature-dropdown", "value"),
    prevent_initial_call=True,  # Don't run on page load initially, let URL set dropdown
)
def update_url_on_dropdown_change(selected_feature_id):
    if (
        selected_feature_id is not None and selected_feature_id != -1
    ):  # -1 used for disabled/no feature states
        return f"?feature={selected_feature_id}"
    return ""
