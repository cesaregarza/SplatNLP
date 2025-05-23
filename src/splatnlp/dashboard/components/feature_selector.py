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
        # Store to trigger dropdown refresh when names change
        dcc.Store(id="feature-names-updated", data=0),
    ],
    className="mb-4",
)


@callback(
    Output("feature-dropdown", "options"),
    Output("feature-dropdown", "value"),
    Input("page-load-trigger", "data"),
    Input(
        "feature-names-updated", "data"
    ),  # Trigger refresh when names are updated
    State("feature-dropdown", "value"),
    State("url", "search"),
)
def populate_feature_options(
    page_load_data: Optional[str],
    names_updated_counter: Optional[
        int
    ],  # Counter that increments when names change
    current_value: Optional[int],
    search_query: Optional[str],
) -> Tuple[List[dict], Optional[int]]:
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Use the counter to ensure Dash detects changes
    print(f"Dropdown refresh triggered, counter: {names_updated_counter}")

    default_value = 0  # Default to feature 0 if no other value is set
    options: List[dict] = []

    if (
        DASHBOARD_CONTEXT is not None
        and hasattr(DASHBOARD_CONTEXT, "sae_model")
        and DASHBOARD_CONTEXT.sae_model is not None
        and hasattr(DASHBOARD_CONTEXT.sae_model, "hidden_dim")
    ):
        num_features = DASHBOARD_CONTEXT.sae_model.hidden_dim
        if num_features > 0:
            # Get feature names if available
            feature_names_manager = getattr(
                DASHBOARD_CONTEXT, "feature_names_manager", None
            )
            options = []
            for i in range(num_features):
                if feature_names_manager:
                    label = feature_names_manager.get_display_name(i)
                else:
                    label = f"Feature {i}"
                options.append({"label": label, "value": i})

            # Debug: print some named features
            if (
                feature_names_manager
                and len(feature_names_manager.feature_names) > 0
            ):
                print(
                    f"Named features: {list(feature_names_manager.feature_names.items())[:5]}"
                )
        else:  # num_features is 0 or less (e.g. model not fully loaded or invalid)
            options = [
                {
                    "label": "No features available",
                    "value": -1,
                    "disabled": True,
                }
            ]
            default_value = -1
    else:
        options = [
            {"label": "Context/Model not loaded", "value": -1, "disabled": True}
        ]
        default_value = -1

    # Determine initial value based on URL, then current_value, then default
    final_value = default_value

    # Prioritize URL query parameter for initial value
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
                # Check if this value is valid among the generated options
                if any(
                    opt["value"] == feature_val_from_url
                    for opt in options
                    if not opt.get("disabled")
                ):
                    final_value = feature_val_from_url
                # If URL param is invalid (e.g., out of range, or refers to a disabled option),
                # final_value remains default_value (which is 0 or -1 based on context)
        except Exception:
            pass  # Ignore parsing errors, final_value remains as determined by context/default

    # If current_value is already set (e.g., by user interaction or previous state) and is valid, it takes precedence
    # over default_value, but not over a valid URL parameter.
    # The order of precedence should be: Valid URL > Valid Current Value > Default from Context > Fallback Default
    # The logic above sets final_value based on URL first, then it will be overwritten if current_value is valid.
    # Let's refine:

    # Start with default based on context
    determined_value = default_value

    # Override with URL if valid
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
                    determined_value = feature_val_from_url
        except Exception:
            pass  # Ignore parsing errors

    # If current_value is valid and different from determined_value (which might be from URL or default),
    # it means user might have interacted. However, on initial load, we typically want URL to be authoritative.
    # For this callback, `current_value` might be `None` or a stale value if triggered by page-load.
    # The main goal here is to set initial state from URL or default.
    # If `current_value` is already valid and `search_query` didn't yield a valid feature, we can consider `current_value`.

    # If determined_value is still the initial default_value (meaning URL didn't provide a valid one),
    # then check if current_value is valid.
    if determined_value == default_value and current_value is not None:
        if any(
            opt["value"] == current_value
            for opt in options
            if not opt.get("disabled")
        ):
            final_value = current_value
        else:
            final_value = determined_value  # current_value is invalid, stick to determined (default or URL)
    else:
        final_value = determined_value  # URL value was valid, or it's the default from context

    # Ensure final_value is actually among the available options if options exist
    if not any(
        opt["value"] == final_value
        for opt in options
        if not opt.get("disabled")
    ):
        if options and not options[0].get("disabled"):
            final_value = options[0][
                "value"
            ]  # Fallback to the first available option
        elif options and options[0].get("disabled"):
            final_value = options[0][
                "value"
            ]  # Fallback to the disabled value like -1
        # If options list is empty (should not happen with current logic), final_value remains as is.

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
