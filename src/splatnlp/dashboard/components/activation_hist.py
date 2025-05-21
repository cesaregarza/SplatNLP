from dash import dcc, html

activation_hist = html.Div(
    [
        html.H4("Activation Histogram"),
        dcc.Graph(
            id="activation-histogram",
            figure={
                "data": [
                    {
                        "x": [0, 1, 2, 3, 4, 5],
                        "y": [10, 20, 15, 5, 2, 1],
                        "type": "bar",
                    }
                ],
                "layout": {
                    "title": "Feature Activation Histogram (placeholder)"
                },
            },
        ),
    ]
)
