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
    import json # For parsing JSON strings from DB if necessary
    import logging # For logging
    logger = logging.getLogger(__name__)

    if selected_feature_id is None:
        return [], "Select a feature."
    
    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, 'db_context') or DASHBOARD_CONTEXT.db_context is None:
        logger.warning("IntervalsGrid: Dashboard context or DB context not available.")
        return [], "Error: Database context not available. Ensure data is loaded correctly."

    db_context = DASHBOARD_CONTEXT.db_context
    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab
    _, id_to_name, _ = generate_maps() # General mapping, should be fine

    try:
        # Fetch pre-binned and pre-sampled data
        binned_samples_data = db_context.get_binned_feature_samples(selected_feature_id)
        if not binned_samples_data:
            return [html.P(f"No binned sample data found for feature {selected_feature_id} in the database.")], ""
    except Exception as e:
        logger.error(f"IntervalsGrid: Failed to get binned feature samples for {selected_feature_id} from DB: {e}", exc_info=True)
        return [], f"Error fetching binned samples: {str(e)}"

    sections: List[Any] = []
    top_tfidf_tokens = set() # TF-IDF analysis is not performed here. Pass empty set.

    # Iterate through the bins (already sorted by bin_index by the DB method)
    # For display, it's often preferred to show highest activation bins first.
    # The current data is sorted by bin_index ASC. If we want DESC, we reverse here.
    for bin_data in reversed(binned_samples_data):
        bin_idx = bin_data["bin_index"]
        min_act = bin_data["bin_min_activation"]
        max_act = bin_data["bin_max_activation"]
        samples_in_bin = bin_data["samples"]
        
        num_examples_in_interval = len(samples_in_bin) # This is the count of *sampled* examples
        
        # Note: The 'count' of total examples originally in this bin before sampling by extract-top-examples
        # is not directly available in `binned_samples_data` structure.
        # If that count is needed, `feature_stats.sampled_histogram_data` might be a source,
        # or `extract-top-examples` output CSV/JSON would need to include original bin counts.
        # For now, we display based on the number of *provided samples* for the bin.
        header_text = f"Bin {bin_idx}: [{min_act:.3f}, {max_act:.3f}) - {num_examples_in_interval} sampled examples"
        header = html.Div(header_text, className="fw-bold mb-2")
        
        card_cols: List[Any] = []
        if num_examples_in_interval > 0:
            for sample in samples_in_bin: # These are already the chosen samples
                example_id_to_fetch = sample["example_id"]
                activation_val_for_card = sample["activation_value"]

                try:
                    # Fetch full example details from DB using its ID
                    with db_context.db.get_connection() as conn:
                        cur = conn.execute("SELECT * FROM examples WHERE id = ?", (example_id_to_fetch,))
                        row = cur.fetchone()
                    
                    if row:
                        example_record_dict = dict(row)
                        # Parse JSON fields if necessary
                        if 'ability_input_tokens' in example_record_dict and isinstance(example_record_dict['ability_input_tokens'], str):
                            example_record_dict['ability_input_tokens'] = json.loads(example_record_dict['ability_input_tokens'])
                        else:
                            example_record_dict['ability_input_tokens'] = example_record_dict.get('ability_input_tokens', [])
                        
                        rec_series = pd.Series(example_record_dict)
                        card = _example_card(
                            rec_series,
                            inv_vocab,
                            inv_weapon_vocab,
                            activation_val_for_card,
                            id_to_name,
                            top_tfidf_tokens, # Currently empty
                        )
                        card_cols.append(dbc.Col(card, width="auto", className="p-1"))
                    else:
                        logger.warning(f"IntervalsGrid: Example ID {example_id_to_fetch} not found in DB.")
                        card_cols.append(dbc.Col(html.P(f"Details for Ex. {example_id_to_fetch} not found."), width="auto"))
                except Exception as card_ex:
                    logger.error(f"IntervalsGrid: Error creating card for example {example_id_to_fetch}: {card_ex}", exc_info=True)
                    card_cols.append(dbc.Col(html.P(f"Error for Ex. {example_id_to_fetch}."), width="auto"))
        else:
            # This case should ideally not happen if get_binned_feature_samples filters out bins with no samples.
            # If it can return bins with an empty "samples" list, this is the correct handling.
            card_cols.append(html.I("No sampled examples in this bin."))

        sections.append(
            html.Div(
                [header, dbc.Row(card_cols, className="g-2 flex-wrap justify-content-start")],
                className="p-3 mb-3 border rounded bg-light",
            )
        )
        
    if not sections: # If binned_samples_data was empty or all bins had no samples
        return [html.P(f"No displayable binned sample data found for feature {selected_feature_id}.") ], ""
        
    return sections, ""
