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
# Helper functions
# ---------------------------------------------------------------------------


def _get_tfidf_color(score: float, items: list) -> str:
    """Get color for TF-IDF badge based on score relative to others."""
    if not items:
        return "primary"

    scores = [s for _, s in items]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return "primary"

    # Normalize score to 0-1 range
    normalized = (score - min_score) / (max_score - min_score)

    # Color gradient from light to dark based on score
    if normalized > 0.8:
        return "danger"  # Red for highest scores
    elif normalized > 0.6:
        return "warning"  # Orange
    elif normalized > 0.4:
        return "primary"  # Blue
    elif normalized > 0.2:
        return "info"  # Light blue
    else:
        return "secondary"  # Gray for lowest scores


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
                html.H6(
                    weapon_name,
                    className="card-title mb-2 text-truncate",
                    title=weapon_name,
                ),
                html.Div(
                    ability_str,
                    className="flex-grow-1 overflow-auto mb-2",
                    style={"fontSize": "0.9rem"},
                ),
                html.P(
                    f"Activation: {activation_val:.4f}",
                    className="mb-0 fw-semibold text-primary",
                ),
            ],
            className="d-flex flex-column",
            style={"height": "180px"},
        ),
        style={"width": "250px", "height": "180px"},
        className="shadow-sm h-100",
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

        # Get weapons that activate this feature most
        top_weapons = []
        if len(top_indices) > 0:
            weapon_counts = {}
            for idx in top_indices:
                wid = int(analysis_df.iloc[idx].get("weapon_id_token", -1))
                raw_wpn = inv_weapon_vocab.get(wid, f"WPN_{wid}")
                weapon_name = id_to_name.get(raw_wpn.split("_")[-1], raw_wpn)
                weapon_counts[weapon_name] = (
                    weapon_counts.get(weapon_name, 0) + 1
                )

            # Get top 5 weapons
            sorted_weapons = sorted(
                weapon_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
            top_weapons = [
                (name, count / len(top_indices))
                for name, count in sorted_weapons
            ]

        if items:
            # Feature name display
            feature_names_manager = getattr(
                DASHBOARD_CONTEXT, "feature_names_manager", None
            )
            if feature_names_manager:
                feature_display = feature_names_manager.get_display_name(
                    selected_feature_id
                )
            else:
                feature_display = f"Feature {selected_feature_id}"

            sections.append(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(
                                f"{feature_display} - Top Activations Analysis",
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Top TF-IDF Tokens",
                                                className="text-muted mb-2",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Badge(
                                                        [tok, f" {score:.2f}"],
                                                        color=_get_tfidf_color(
                                                            score, items
                                                        ),
                                                        className="me-2 mb-2",
                                                        pill=True,
                                                        style={
                                                            "fontSize": "0.9rem"
                                                        },
                                                    )
                                                    for tok, score in items
                                                ],
                                                className="d-flex flex-wrap",
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    (
                                        dbc.Col(
                                            [
                                                html.H6(
                                                    "Top Activating Weapons",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            weapon,
                                                                            className="me-2",
                                                                        ),
                                                                        html.Small(
                                                                            f"({pct:.0%})",
                                                                            className="text-muted",
                                                                        ),
                                                                    ],
                                                                    className="d-flex justify-content-between",
                                                                ),
                                                                dbc.Progress(
                                                                    value=pct
                                                                    * 100,
                                                                    color="success",
                                                                    className="mb-1",
                                                                    style={
                                                                        "height": "15px"
                                                                    },
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        )
                                                        for weapon, pct in top_weapons
                                                    ]
                                                ),
                                            ],
                                            md=6,
                                        )
                                        if top_weapons
                                        else html.Div()
                                    ),
                                ]
                            ),
                        ]
                    ),
                    className="mb-4 shadow-sm",
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
