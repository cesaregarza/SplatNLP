import random
from typing import Any, List, Optional

import dash_bootstrap_components as dbc
import h5py
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html
from sklearn.feature_extraction.text import TfidfVectorizer

from splatnlp.preprocessing.transform.mappings import generate_maps

# ---------------------------------------------------------------------------
# Helper - compact card (no tooltips)
# ---------------------------------------------------------------------------


def _example_card(
    record: pd.Series,
    inv_vocab: dict[str, str],
    inv_weapon_vocab: dict[int, str],
    activation_val: float,
    id_to_name: dict[str, str],
    top_tfidf_tokens: set[str],
) -> dbc.Card:
    """Return a small bootstrap card summarising an example (no hover projections)."""

    ability_ids: list[int] = record.get("ability_input_tokens", [])
    ability_names = [
        inv_vocab.get(str(t), inv_vocab.get(t, f"ID_{t}"))
        for t in ability_ids
        if inv_vocab.get(str(t), inv_vocab.get(t, f"ID_{t}"))
        not in ("<PAD>", "<NULL>")
    ]

    # Format ability names with bold for top TF-IDF tokens
    formatted_names = []
    for name in ability_names:
        if name in top_tfidf_tokens:
            formatted_names.append(html.Strong(name))
        else:
            formatted_names.append(name)

    ability_str = (
        html.Span(
            [html.Span(name, className="me-1") for name in formatted_names]
        )
        or "N/A"
    )

    wid = int(record.get("weapon_id_token", -1))
    raw_wpn = inv_weapon_vocab.get(wid, f"WPN_{wid}")
    weapon_name = id_to_name.get(raw_wpn.split("_")[-1], raw_wpn)

    card = dbc.Card(
        dbc.CardBody(
            [
                html.H6(weapon_name, className="card-title mb-2"),
                html.P(ability_str, className="small mb-2"),
                html.P(
                    f"Activation: {activation_val:.4f}",
                    className="mb-0 fw-semibold",
                ),
            ]
        ),
        style={"width": "220px"},
        className="shadow-sm",
    )
    return card


# ---------------------------------------------------------------------------
# Layout container
# ---------------------------------------------------------------------------

intervals_grid_component = html.Div(
    [
        html.H4("Subsampled Intervals Grid", className="mb-3"),
        dcc.Loading(
            id="loading-intervals-grid",
            type="default",
            children=html.Div(id="intervals-grid-display"),
        ),
        html.P(id="intervals-grid-error-message", style={"color": "red"}),
    ],
    id="intervals-grid-content",
    className="mb-4",
)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@callback(
    [
        Output("intervals-grid-display", "children"),
        Output("intervals-grid-error-message", "children"),
    ],
    Input("feature-dropdown", "value"),
)
def render_intervals_grid(selected_feature_id: Optional[int]):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None or DASHBOARD_CONTEXT is None:
        return [], "Select a feature."

    acts_matrix = DASHBOARD_CONTEXT.all_sae_hidden_activations
    if not (0 <= selected_feature_id < acts_matrix.shape[1]):
        return [], f"Feature {selected_feature_id} out of range."

    analysis_df: pd.DataFrame = DASHBOARD_CONTEXT.analysis_df_records
    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab
    _, id_to_name, _ = generate_maps()

    acts = acts_matrix[:, selected_feature_id]
    lo, hi = float(np.min(acts)), float(np.max(acts) + 1e-6)
    bins = 10
    bounds = np.linspace(lo, hi, bins + 1)
    per_interval = 5

    sections: List[Any] = []

    # Get top TF-IDF tokens first ---------------------------------------------------------
    top_tfidf_tokens = set()
    try:
        top_k = min(len(acts), 100)
        top_indices = np.argsort(acts)[-top_k:]

        def to_doc(token_ids: list[int]):
            return " ".join(
                inv_vocab.get(str(t), inv_vocab.get(t, str(t)))
                for t in token_ids
            )

        corpus_all = [
            to_doc(lst) for lst in analysis_df["ability_input_tokens"].tolist()
        ]
        corpus_top = [
            to_doc(analysis_df.iloc[i]["ability_input_tokens"])
            for i in top_indices
        ]
        vec = TfidfVectorizer(min_df=2, max_df=0.95, token_pattern=r"\S+").fit(
            corpus_all
        )
        mat = vec.transform(corpus_top)
        avg_scores = np.asarray(mat.sum(axis=0)).ravel() / top_k
        best = np.argsort(avg_scores)[-10:][::-1]
        items = [
            (vec.get_feature_names_out()[i], avg_scores[i])
            for i in best
            if avg_scores[i] > 0
        ]
        top_tfidf_tokens = {tok for tok, _ in items}

        if items:
            sections.append(
                html.Div(
                    [
                        html.H5(
                            "Top TF-IDF tokens (top-100 activations)",
                            className="mb-3",
                        ),
                        html.Ul(
                            [
                                html.Li([html.Strong(tok), f": {score:.3f}"])
                                for tok, score in items
                            ]
                        ),
                    ]
                )
            )
    except Exception as e:
        sections.append(
            html.P(f"TF-IDF error: {str(e)[:80]}", className="text-danger")
        )

    # Iterate **high -> low**
    for bin_idx in reversed(range(bins)):
        a, b = bounds[bin_idx], bounds[bin_idx + 1]
        mask = (acts >= a) & (acts < b if bin_idx < bins - 1 else acts <= b)
        idxs = np.where(mask)[0]

        header = html.Div(
            f"Interval {bins - bin_idx}: [{a:.3f}, {b:.3f}) - {len(idxs)} exs",
            className="fw-bold mb-2",
        )

        card_cols: List[Any] = []
        if idxs.size:
            chosen = random.sample(list(idxs), min(len(idxs), per_interval))
            for ex_idx in chosen:
                rec = analysis_df.iloc[ex_idx]
                card = _example_card(
                    rec,
                    inv_vocab,
                    inv_weapon_vocab,
                    float(acts[ex_idx]),
                    id_to_name,
                    top_tfidf_tokens,
                )
                card_cols.append(dbc.Col(card, width="auto", className="p-1"))
        else:
            card_cols.append(html.I("No examples in this interval."))

        sections.append(
            html.Div(
                [
                    header,
                    dbc.Row(
                        card_cols,
                        className="g-2 flex-wrap justify-content-start",
                    ),
                ],
                className="p-3 mb-3 border rounded bg-light",
            )
        )

    return sections, ""
