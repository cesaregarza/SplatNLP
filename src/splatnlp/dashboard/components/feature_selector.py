from dash import dcc, html

# Placeholder feature list (replace with real feature indices later)
feature_options = [{"label": f"Feature {i}", "value": i} for i in range(10)]

feature_selector = html.Div(
    [
        html.Label("Select SAE Feature:"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=feature_options,
            value=0,  # default selection
            clearable=False,
        ),
    ]
)
