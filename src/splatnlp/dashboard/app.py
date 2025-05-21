import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from .components.activation_hist import activation_hist

# Import components (to be created)
from .components.feature_selector import feature_selector

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("SAE Feature Dashboard"),
                        feature_selector,
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Div(id="feature-summary"),
                        activation_hist,
                        html.Div(id="logits-table"),
                        html.Div(id="examples-table"),
                        html.Div(id="intervals-grid"),
                        html.Div(id="correlations-table"),
                    ],
                    width=9,
                ),
            ]
        )
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)
