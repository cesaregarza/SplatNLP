"""Components for token analysis in the dashboard."""

import dash_bootstrap_components as dbc
from dash import html


def create_single_token_component():
    """Create component for displaying single token examples."""
    return html.Div(
        [
            html.H4("Single Token Examples"),
            html.Div(id="single-token-examples"),
        ]
    )


def create_token_pair_component():
    """Create component for displaying token pair examples."""
    return html.Div(
        [
            html.H4("Token Pair Examples"),
            html.Div(id="token-pair-examples"),
        ]
    )


def create_token_triple_component():
    """Create component for displaying token triple examples."""
    return html.Div(
        [
            html.H4("Token Triple Examples"),
            html.Div(id="token-triple-examples"),
        ]
    )


def create_token_analysis_tab():
    """Create the token analysis tab with all components."""
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Single Tokens",
                        children=create_single_token_component(),
                    ),
                    dbc.Tab(
                        label="Token Pairs",
                        children=create_token_pair_component(),
                    ),
                    dbc.Tab(
                        label="Token Triples",
                        children=create_token_triple_component(),
                    ),
                ]
            )
        ]
    )
