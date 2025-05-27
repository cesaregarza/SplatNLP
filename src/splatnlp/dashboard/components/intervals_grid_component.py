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
        acts_values, example_ids_for_acts = db_context.get_feature_activations(selected_feature_id)
        if acts_values.size == 0:
            return [html.P("No activation data found for this feature in the database.")], ""
    except Exception as e:
        logger.error(f"IntervalsGrid: Failed to get feature activations for {selected_feature_id} from DB: {e}", exc_info=True)
        return [], f"Error fetching activations: {str(e)}"

    lo, hi = float(np.min(acts_values)), float(np.max(acts_values) + 1e-6) # Ensure hi includes max value
    bins = 10
    bounds = np.linspace(lo, hi, bins + 1)
    per_interval = 5
    sections: List[Any] = []

    # TF-IDF and Top Weapons analysis is removed when using db_context,
    # as it would require fetching all example texts, which is inefficient.
    # This kind of global analysis should be precomputed and stored in the DB if needed.
    top_tfidf_tokens = set() # Pass empty set to _example_card

    # Iterate **high -> low** for bins
    for bin_idx in reversed(range(bins)):
        a, b = bounds[bin_idx], bounds[bin_idx + 1]
        # Ensure the last bin correctly includes the max value if it's exactly on the boundary
        mask = (acts_values >= a) & (acts_values < b if bin_idx < bins - 1 else acts_values <= b)
        
        # Indices relative to the `acts_values` array
        current_interval_indices_in_acts_array = np.where(mask)[0] 
        
        num_examples_in_interval = len(current_interval_indices_in_acts_array)
        header_text = f"Interval {bins - bin_idx}: [{a:.3f}, {b:.3f}) - {num_examples_in_interval} examples"
        header = html.Div(header_text, className="fw-bold mb-2")
        
        card_cols: List[Any] = []
        if num_examples_in_interval > 0:
            num_to_sample = min(num_examples_in_interval, per_interval)
            # Get indices within the `current_interval_indices_in_acts_array` for sampling
            chosen_indices_for_sampling = np.random.choice(num_examples_in_interval, num_to_sample, replace=False)
            
            for chosen_idx in chosen_indices_for_sampling:
                # Map back to the index in the original `acts_values` and `example_ids_for_acts`
                original_array_idx = current_interval_indices_in_acts_array[chosen_idx]
                example_id_to_fetch = int(example_ids_for_acts[original_array_idx]) # Ensure it's Python int
                activation_val_for_card = float(acts_values[original_array_idx])

                try:
                    # Fetch full example details from DB using its ID
                    # This assumes db_context has a method like get_example_details,
                    # or we build the query here.
                    with db_context.db.get_connection() as conn: # Access underlying DashboardDatabase instance
                        cur = conn.execute("SELECT * FROM examples WHERE id = ?", (example_id_to_fetch,))
                        row = cur.fetchone()
                    
                    if row:
                        example_record_dict = dict(row)
                        # Parse JSON fields if necessary
                        if 'ability_input_tokens' in example_record_dict and isinstance(example_record_dict['ability_input_tokens'], str):
                            example_record_dict['ability_input_tokens'] = json.loads(example_record_dict['ability_input_tokens'])
                        else: # Ensure it's a list
                            example_record_dict['ability_input_tokens'] = example_record_dict.get('ability_input_tokens', [])
                        
                        # _example_card expects a pd.Series like object
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
            card_cols.append(html.I("No examples in this interval."))

        sections.append(
            html.Div(
                [header, dbc.Row(card_cols, className="g-2 flex-wrap justify-content-start")],
                className="p-3 mb-3 border rounded bg-light",
            )
        )
    return sections, ""
