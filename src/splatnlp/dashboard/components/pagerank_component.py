"""PageRank analysis component for the dashboard."""

import logging
from collections import Counter

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from splatnlp.dashboard.utils.converters import get_weapon_properties_df
from splatnlp.preprocessing.transform.mappings import generate_maps
from splatnlp.utils.constants import MAIN_ONLY_ABILITIES

logger = logging.getLogger(__name__)


def _make_chart_row(row_id: str, title: str, description: str) -> dbc.Row:
    """Create a row of 3 charts for a PageRank analysis type."""
    return html.Div(
        [
            html.H5(title, className="mb-2 mt-3"),
            html.P(description, className="text-muted small mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Raw Tokens", className="mb-1 text-center"),
                            html.P(
                                "Weight = 1",
                                className="text-muted small text-center mb-1",
                            ),
                            dcc.Graph(
                                id=f"pagerank-chart-{row_id}-raw",
                                style={"height": "280px"},
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.H6(
                                "Weight by AP", className="mb-1 text-center"
                            ),
                            html.P(
                                "Weight = AP",
                                className="text-muted small text-center mb-1",
                            ),
                            dcc.Graph(
                                id=f"pagerank-chart-{row_id}-ap",
                                style={"height": "280px"},
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.H6(
                                "Family Mode", className="mb-1 text-center"
                            ),
                            html.P(
                                "Collapsed, max AP",
                                className="text-muted small text-center mb-1",
                            ),
                            dcc.Graph(
                                id=f"pagerank-chart-{row_id}-family",
                                style={"height": "280px"},
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
        ]
    )


def _make_grid_tab(tab_id: str, label: str, columns: list) -> dbc.Tab:
    """Create a tab with an AgGrid for PageRank results."""
    return dbc.Tab(
        label=label,
        tab_id=tab_id,
        children=[
            dag.AgGrid(
                id=f"pagerank-grid-{tab_id.replace('tab-pr-', '')}",
                rowData=[],
                columnDefs=columns,
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                },
                dashGridOptions={"domLayout": "normal"},
                style={"height": "250px", "width": "100%"},
                className="mt-2",
            ),
        ],
    )


# Column definitions for different grid types
ABILITY_COLS = [
    {"field": "Rank", "width": 70},
    {"field": "Token", "width": 180},
    {"field": "Score", "width": 120},
    {"field": "AP", "width": 80},
]
FAMILY_COLS = [
    {"field": "Rank", "width": 70},
    {"field": "Family", "width": 180},
    {"field": "Score", "width": 120},
    {"field": "Type", "width": 100},
]
COMPOUND_COLS = [
    {"field": "Rank", "width": 60},
    {"field": "Category", "width": 140},
    {"field": "Ability", "width": 160},
    {"field": "Score", "width": 100},
    {"field": "Type", "width": 90},
]

pagerank_component = html.Div(
    id="pagerank-content",
    children=[
        html.H4("PageRank Token Analysis", className="mb-3"),
        html.P(
            "Analyzes token importance using PageRank on co-occurrence graphs. "
            "Compound analyses (weapon/sub/special/class) are normalized by 1/n.",
            className="text-muted small mb-3",
        ),
        # Run button and copy
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Run PageRank (All Modes)",
                            id="run-pagerank-btn",
                            color="primary",
                            size="lg",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Clipboard(
                                    target_id="pagerank-markdown-content",
                                    title="Copy to clipboard (Markdown)",
                                    className="btn btn-outline-secondary",
                                    style={"fontSize": "1.2rem"},
                                ),
                                html.Small(
                                    "Copy MD",
                                    className="d-block text-muted mt-1",
                                ),
                            ],
                            className="text-center",
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Checkbox(
                            id="pagerank-truncate-checkbox",
                            label="Truncate MD (15 rows)",
                            value=False,
                        ),
                    ],
                    md=2,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    [
                        html.Div(id="pagerank-status", className="mb-0"),
                    ],
                    md=7,
                ),
            ],
            className="mb-4 p-3 border rounded bg-light align-items-center",
        ),
        # Hidden div for markdown content
        html.Div("", id="pagerank-markdown-content", style={"display": "none"}),
        # Results
        dcc.Loading(
            id="loading-pagerank",
            type="default",
            children=[
                # Row 1: Ability-only
                _make_chart_row(
                    "ability",
                    "Ability-Only Co-occurrence",
                    "Pure ability token analysis",
                ),
                # Row 2: By Weapon
                _make_chart_row(
                    "weapon",
                    "By Weapon",
                    "PageRank on Weapon + Ability compound tokens",
                ),
                # Row 3: By Sub (rollup)
                _make_chart_row(
                    "sub",
                    "By Sub Weapon (rollup)",
                    "Weapon scores rolled up by sub, normalized by 1/n",
                ),
                # Row 4: By Special (rollup)
                _make_chart_row(
                    "special",
                    "By Special (rollup)",
                    "Weapon scores rolled up by special, normalized by 1/n",
                ),
                # Tables with tabs
                dbc.Tabs(
                    [
                        # Ability tabs
                        _make_grid_tab(
                            "tab-pr-ability-raw", "Ability Raw", ABILITY_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-ability-ap", "Ability AP", ABILITY_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-ability-family",
                            "Ability Family",
                            FAMILY_COLS,
                        ),
                        # Weapon tabs
                        _make_grid_tab(
                            "tab-pr-weapon-raw", "Weapon Raw", COMPOUND_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-weapon-ap", "Weapon AP", COMPOUND_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-weapon-family",
                            "Weapon Family",
                            COMPOUND_COLS,
                        ),
                        # Sub tabs
                        _make_grid_tab(
                            "tab-pr-sub-raw", "Sub Raw", COMPOUND_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-sub-ap", "Sub AP", COMPOUND_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-sub-family", "Sub Family", COMPOUND_COLS
                        ),
                        # Special tabs
                        _make_grid_tab(
                            "tab-pr-special-raw", "Special Raw", COMPOUND_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-special-ap", "Special AP", COMPOUND_COLS
                        ),
                        _make_grid_tab(
                            "tab-pr-special-family",
                            "Special Family",
                            COMPOUND_COLS,
                        ),
                    ],
                    id="pagerank-tabs",
                    active_tab="tab-pr-ability-raw",
                ),
            ],
        ),
        html.P(id="pagerank-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


def get_family_type(family_name: str) -> str:
    """Determine if a family is Standard or Main Only."""
    if family_name in MAIN_ONLY_ABILITIES:
        return "Main Only"
    return "Standard"


def get_compound_token_type(token: str) -> str:
    """Determine the type of a compound token."""
    import re

    for main_only in MAIN_ONLY_ABILITIES:
        if f"_{main_only}" in token or token.endswith(main_only):
            return "Main Only"
    if re.search(r"_\d+$", token):
        return "Standard"
    return "Compound"


def parse_compound_token(
    token: str, category_to_name: dict = None
) -> tuple[str, str]:
    """Parse a compound token into (category_name, ability_name).

    Tokens are in format: CategoryName_ability_name_XX
    We find the longest matching category prefix.
    """
    if not category_to_name:
        # Can't parse without mapping
        return "", token

    # Try to find longest matching category prefix
    best_match = None
    best_length = 0

    for category_name in category_to_name.values():
        # Clean category name for matching (same as when building tokens)
        cleaned = (
            category_name.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("'", "")
        )
        if token.startswith(cleaned + "_"):
            if len(cleaned) > best_length:
                best_match = category_name
                best_length = len(cleaned)

    if best_match:
        ability_part = token[best_length + 1 :]
        return best_match, ability_part

    return "", token


def format_pagerank_markdown(
    feature_id: int,
    feature_name: str,
    n_examples: int,
    results: dict,
    category_mappings: dict = None,
    truncate: bool = False,
) -> str:
    """Format PageRank results as Markdown for clipboard export.

    Args:
        truncate: If True, export 15 rows per section instead of 30.
    """
    n_rows = 15 if truncate else 30

    lines = [
        f"# PageRank Analysis: Feature {feature_id}",
        "",
        f"**Feature:** {feature_name}",
        f"**Examples Processed:** {n_examples:,}",
        "",
    ]

    for section, section_results in results.items():
        is_compound = section != "Ability-Only Analysis"
        lines.append(f"## {section}")
        lines.append("")

        # Get category mapping for this section
        cat_mapping = None
        if category_mappings and is_compound:
            for key in ["weapon", "sub", "special", "class"]:
                if key.lower() in section.lower():
                    cat_mapping = category_mappings.get(key)
                    break

        for mode_label, top_tokens in section_results:
            if is_compound and cat_mapping:
                lines.extend(
                    [
                        f"### {mode_label}",
                        "",
                        "| Rank | Category | Ability | Score |",
                        "|------|----------|---------|-------|",
                    ]
                )
                for i, (token, _, score) in enumerate(top_tokens[:n_rows], 1):
                    cat, ability = parse_compound_token(token, cat_mapping)
                    lines.append(f"| {i} | {cat} | {ability} | {score:.6f} |")
            else:
                lines.extend(
                    [
                        f"### {mode_label}",
                        "",
                        "| Rank | Token | Score |",
                        "|------|-------|-------|",
                    ]
                )
                for i, (token, _, score) in enumerate(top_tokens[:n_rows], 1):
                    lines.append(f"| {i} | {token} | {score:.6f} |")
            lines.append("")

    return "\n".join(lines)


def create_bar_chart(top_tokens: list, title: str, color: str) -> go.Figure:
    """Create a bar chart for PageRank results."""
    if not top_tokens:
        fig = go.Figure()
        fig.update_layout(
            title="No data", xaxis={"visible": False}, yaxis={"visible": False}
        )
        return fig

    # Truncate long labels
    labels = []
    for t in top_tokens[:12]:
        label = t[0]
        if len(label) > 25:
            label = label[:22] + "..."
        labels.append(label)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[t[2] for t in top_tokens[:12]],
            marker_color=color,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        xaxis_tickangle=-45,
        margin={"l": 40, "r": 10, "t": 35, "b": 100},
        yaxis_title="Score",
    )
    return fig


def build_weapon_compound_analysis(
    examples_list: list,
    inv_vocab: dict,
    weapon_to_name: dict,
) -> tuple[dict, dict, list]:
    """Build weapon compound tokens and examples.

    Returns: (compound_vocab, compound_inv_vocab, examples_with_compounds)
    """
    compound_token_set = set()

    # First pass: collect unique compound tokens
    for row in examples_list:
        weapon_id = row["weapon_id"]
        weapon_name = weapon_to_name.get(weapon_id)
        if not weapon_name:
            continue

        # Clean weapon name for token
        weapon_prefix = (
            weapon_name.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("'", "")
        )

        for ability_id in row["ability_input_tokens"]:
            ability_name = inv_vocab.get(ability_id, f"ability_{ability_id}")
            if not ability_name.startswith("<"):
                compound_token = f"{weapon_prefix}_{ability_name}"
                compound_token_set.add(compound_token)

    # Build vocab
    compound_vocab = {
        token: idx for idx, token in enumerate(sorted(compound_token_set))
    }
    compound_inv_vocab = {idx: token for token, idx in compound_vocab.items()}

    # Second pass: build examples (no normalization for weapons - each is unique)
    examples_with_compounds = []
    for row in examples_list:
        weapon_id = row["weapon_id"]
        weapon_name = weapon_to_name.get(weapon_id)
        if not weapon_name:
            continue

        weapon_prefix = (
            weapon_name.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("'", "")
        )

        compound_ids = []
        for ability_id in row["ability_input_tokens"]:
            ability_name = inv_vocab.get(ability_id, f"ability_{ability_id}")
            if not ability_name.startswith("<"):
                compound_token = f"{weapon_prefix}_{ability_name}"
                if compound_token in compound_vocab:
                    compound_ids.append(compound_vocab[compound_token])

        if compound_ids:
            examples_with_compounds.append(
                {
                    "tokens": compound_ids,
                    "activation": row["activation"],
                    "weapon_id": weapon_id,
                }
            )

    return compound_vocab, compound_inv_vocab, examples_with_compounds


def rollup_weapon_scores(
    weapon_results: list,
    weapon_to_name: dict,
    weapon_to_category: dict,
    category_counts: Counter,
) -> list:
    """Roll up weapon PageRank scores to a category (sub/special/class).

    Takes weapon results like [(weapon_ability_token, idx, score), ...]
    and aggregates by category, normalizing by 1/n weapons in category.

    Returns: [(category_ability_token, 0, score), ...]
    """
    from collections import defaultdict

    # Pre-build cleaned weapon name lookup (find longest match)
    cleaned_weapons = []
    for weapon_id, weapon_name in weapon_to_name.items():
        cleaned = (
            weapon_name.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("'", "")
        )
        cleaned_weapons.append((cleaned, weapon_id, len(cleaned)))
    # Sort by length descending so we match longest prefix first
    cleaned_weapons.sort(key=lambda x: x[2], reverse=True)

    # Aggregate scores by category_ability
    category_scores = defaultdict(float)

    for weapon_token, _, score in weapon_results:
        # Parse weapon_ability token to get weapon name and ability
        # Token format: WeaponName_ability_name_XX
        # Find the LONGEST matching weapon prefix
        matched_weapon = None
        ability_part = None

        for cleaned, weapon_id, _ in cleaned_weapons:
            if weapon_token.startswith(cleaned + "_"):
                matched_weapon = weapon_id
                ability_part = weapon_token[len(cleaned) + 1 :]
                break  # Already sorted by length, so first match is longest

        if matched_weapon is None or ability_part is None:
            continue

        # Get category for this weapon
        category = weapon_to_category.get(matched_weapon)
        if not category:
            continue

        # Clean category name
        cat_prefix = (
            category.replace(" ", "_")
            .replace("-", "_")
            .replace(".", "")
            .replace("'", "")
        )

        # Normalize by 1/n weapons in this category
        n_weapons = category_counts.get(category, 1)
        normalized_score = score / n_weapons

        # Aggregate
        category_token = f"{cat_prefix}_{ability_part}"
        category_scores[category_token] += normalized_score

    # Sort by score descending
    sorted_results = sorted(
        category_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Return in same format as PageRank results: (token, idx, score)
    return [(token, 0, score) for token, score in sorted_results[:30]]


@callback(
    [
        Output("pagerank-status", "children"),
        # Ability charts
        Output("pagerank-chart-ability-raw", "figure"),
        Output("pagerank-chart-ability-ap", "figure"),
        Output("pagerank-chart-ability-family", "figure"),
        # Weapon charts
        Output("pagerank-chart-weapon-raw", "figure"),
        Output("pagerank-chart-weapon-ap", "figure"),
        Output("pagerank-chart-weapon-family", "figure"),
        # Sub charts
        Output("pagerank-chart-sub-raw", "figure"),
        Output("pagerank-chart-sub-ap", "figure"),
        Output("pagerank-chart-sub-family", "figure"),
        # Special charts
        Output("pagerank-chart-special-raw", "figure"),
        Output("pagerank-chart-special-ap", "figure"),
        Output("pagerank-chart-special-family", "figure"),
        # Ability grids
        Output("pagerank-grid-ability-raw", "rowData"),
        Output("pagerank-grid-ability-ap", "rowData"),
        Output("pagerank-grid-ability-family", "rowData"),
        # Weapon grids
        Output("pagerank-grid-weapon-raw", "rowData"),
        Output("pagerank-grid-weapon-ap", "rowData"),
        Output("pagerank-grid-weapon-family", "rowData"),
        # Sub grids
        Output("pagerank-grid-sub-raw", "rowData"),
        Output("pagerank-grid-sub-ap", "rowData"),
        Output("pagerank-grid-sub-family", "rowData"),
        # Special grids
        Output("pagerank-grid-special-raw", "rowData"),
        Output("pagerank-grid-special-ap", "rowData"),
        Output("pagerank-grid-special-family", "rowData"),
        # Error and markdown
        Output("pagerank-error-message", "children"),
        Output("pagerank-markdown-content", "children"),
    ],
    Input("run-pagerank-btn", "n_clicks"),
    [
        State("feature-dropdown", "value"),
        State("active-tab-store", "data"),
        State("pagerank-truncate-checkbox", "value"),
    ],
    prevent_initial_call=True,
)
def run_pagerank_analysis(n_clicks, feature_id, active_tab, truncate_mode):
    """Run PageRank analysis on the selected feature for all modes and categories."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    from splatnlp.dashboard.utils.pagerank import PageRankAnalyzer

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No data", xaxis={"visible": False}, yaxis={"visible": False}
    )

    # 1 status + 12 charts + 12 grids + 2 (error, markdown) = 27 outputs
    empty_result = tuple(
        [dbc.Alert("Select a feature first.", color="warning")]
        + [empty_fig] * 12
        + [[]] * 12
        + ["", ""]
    )

    if feature_id is None:
        return empty_result

    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, "db"):
        return tuple(
            [dbc.Alert("Database not available.", color="danger")]
            + [empty_fig] * 12
            + [[]] * 12
            + ["Dashboard context not initialized", ""]
        )

    db = DASHBOARD_CONTEXT.db
    vocab = DASHBOARD_CONTEXT.vocab
    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    # Get mappings
    _, id_to_name, _ = generate_maps()

    # Get weapon properties for sub/special/class lookups
    weapon_props_df = get_weapon_properties_df()

    # Build lookups: weapon_id (int) -> category name
    weapon_to_sub = {}
    weapon_to_special = {}
    weapon_to_name = {}

    for row in weapon_props_df.to_dicts():
        # weapon_id is like "weapon_id_40", extract the number
        wid_str = row["weapon_id"].replace("weapon_id_", "")
        try:
            wid = int(wid_str)
            weapon_name = id_to_name.get(wid_str, f"Weapon {wid}")
            # Skip Replica weapons
            if "Replica" in weapon_name:
                continue
            weapon_to_sub[wid] = row["sub"]
            weapon_to_special[wid] = row["special"]
            weapon_to_name[wid] = weapon_name
        except ValueError:
            continue

    # Count weapons per category for normalization
    sub_counts = Counter(weapon_to_sub.values())
    special_counts = Counter(weapon_to_special.values())

    # Get feature name
    feature_name = f"Feature {feature_id}"
    if hasattr(DASHBOARD_CONTEXT, "feature_labels_manager"):
        label = DASHBOARD_CONTEXT.feature_labels_manager.get_label(feature_id)
        if label and label.name:
            feature_name = f"{feature_id}: {label.name}"

    try:
        logger.info(f"PageRank: Getting activations for feature {feature_id}")
        examples_df = db.get_all_feature_activations_for_pagerank(
            feature_id, include_negative=True
        )

        if len(examples_df) == 0:
            return tuple(
                [dbc.Alert("No examples found for this feature.", color="info")]
                + [empty_fig] * 12
                + [[]] * 12
                + ["", ""]
            )

        n_examples = len(examples_df)
        examples_list = examples_df.to_dicts()
        logger.info(f"PageRank: Building graphs from {n_examples} examples")

        modes = ["raw", "ap_weighted", "family"]
        mode_labels = {
            "raw": "Raw Tokens",
            "ap_weighted": "Weight by AP",
            "family": "Family Mode",
        }
        colors = {
            "raw": "steelblue",
            "ap_weighted": "forestgreen",
            "family": "darkorange",
        }

        # ========== Ability-only analysis ==========
        ability_results = {}
        for mode in modes:
            analyzer = PageRankAnalyzer(vocab, inv_vocab, mode=mode)
            for row in examples_list:
                analyzer.add_example(
                    row["ability_input_tokens"], row["activation"]
                )
            scores = analyzer.compute_pagerank()
            top_tokens = analyzer.get_top_tokens(scores, top_k=30)
            ability_results[mode] = (mode_labels[mode], top_tokens, analyzer)

        # ========== Weapon compound analysis ==========
        weapon_vocab, weapon_inv_vocab, weapon_examples = (
            build_weapon_compound_analysis(
                examples_list, inv_vocab, weapon_to_name
            )
        )

        weapon_results = {}
        weapon_all_scores = {}  # Store ALL scores for rollup (not just top 30)
        for mode in modes:
            analyzer = PageRankAnalyzer(
                weapon_vocab, weapon_inv_vocab, mode=mode
            )
            for row in weapon_examples:
                analyzer.add_example(row["tokens"], row["activation"])
            scores = analyzer.compute_pagerank()
            top_tokens = analyzer.get_top_tokens(scores, top_k=30)
            # Get ALL tokens for rollup
            all_tokens = analyzer.get_top_tokens(scores, top_k=len(scores))
            weapon_results[mode] = (mode_labels[mode], top_tokens, analyzer)
            weapon_all_scores[mode] = all_tokens

        # ========== Roll up weapon scores to sub/special ==========
        # Sub rollup (using ALL weapon scores)
        sub_results = {}
        for mode in modes:
            all_weapon_tokens = weapon_all_scores[mode]
            rolled_up = rollup_weapon_scores(
                all_weapon_tokens, weapon_to_name, weapon_to_sub, sub_counts
            )
            sub_results[mode] = (mode_labels[mode], rolled_up, None)

        # Special rollup (using ALL weapon scores)
        special_results = {}
        for mode in modes:
            all_weapon_tokens = weapon_all_scores[mode]
            rolled_up = rollup_weapon_scores(
                all_weapon_tokens,
                weapon_to_name,
                weapon_to_special,
                special_counts,
            )
            special_results[mode] = (mode_labels[mode], rolled_up, None)

        # ========== Build charts ==========
        # Ability charts
        fig_ability_raw = create_bar_chart(
            ability_results["raw"][1], "Ability Raw", colors["raw"]
        )
        fig_ability_ap = create_bar_chart(
            ability_results["ap_weighted"][1],
            "Ability AP",
            colors["ap_weighted"],
        )
        fig_ability_family = create_bar_chart(
            ability_results["family"][1], "Ability Family", colors["family"]
        )

        # Weapon charts
        fig_weapon_raw = create_bar_chart(
            weapon_results["raw"][1], "Weapon Raw", colors["raw"]
        )
        fig_weapon_ap = create_bar_chart(
            weapon_results["ap_weighted"][1], "Weapon AP", colors["ap_weighted"]
        )
        fig_weapon_family = create_bar_chart(
            weapon_results["family"][1], "Weapon Family", colors["family"]
        )

        # Sub charts
        fig_sub_raw = create_bar_chart(
            sub_results["raw"][1], "Sub Raw", colors["raw"]
        )
        fig_sub_ap = create_bar_chart(
            sub_results["ap_weighted"][1], "Sub AP", colors["ap_weighted"]
        )
        fig_sub_family = create_bar_chart(
            sub_results["family"][1], "Sub Family", colors["family"]
        )

        # Special charts
        fig_special_raw = create_bar_chart(
            special_results["raw"][1], "Special Raw", colors["raw"]
        )
        fig_special_ap = create_bar_chart(
            special_results["ap_weighted"][1],
            "Special AP",
            colors["ap_weighted"],
        )
        fig_special_family = create_bar_chart(
            special_results["family"][1], "Special Family", colors["family"]
        )

        # ========== Build grids ==========
        def build_ability_grid(results_tuple):
            grid = []
            label, top_tokens, analyzer = results_tuple
            for i, (token, token_id, score) in enumerate(top_tokens, 1):
                ap_value = analyzer._ap_cache.get(token_id, 0)
                grid.append(
                    {
                        "Rank": i,
                        "Token": token,
                        "Score": f"{score:.6f}",
                        "AP": ap_value if ap_value > 0 else "-",
                    }
                )
            return grid

        def build_family_grid(results_tuple):
            grid = []
            label, top_tokens, analyzer = results_tuple
            for i, (family, _, score) in enumerate(top_tokens, 1):
                grid.append(
                    {
                        "Rank": i,
                        "Family": family,
                        "Score": f"{score:.6f}",
                        "Type": get_family_type(family),
                    }
                )
            return grid

        def build_compound_grid(results_tuple, cat_mapping):
            grid = []
            label, top_tokens, analyzer = results_tuple
            for i, (token, _, score) in enumerate(top_tokens, 1):
                cat, ability = parse_compound_token(token, cat_mapping)
                grid.append(
                    {
                        "Rank": i,
                        "Category": cat,
                        "Ability": ability,
                        "Score": f"{score:.6f}",
                        "Type": get_compound_token_type(token),
                    }
                )
            return grid

        # Ability grids
        grid_ability_raw = build_ability_grid(ability_results["raw"])
        grid_ability_ap = build_ability_grid(ability_results["ap_weighted"])
        grid_ability_family = build_family_grid(ability_results["family"])

        # Category name mappings for parsing compound tokens
        weapon_name_map = {v: v for v in weapon_to_name.values()}
        sub_name_map = {v: v for v in weapon_to_sub.values()}
        special_name_map = {v: v for v in weapon_to_special.values()}

        # Weapon grids
        grid_weapon_raw = build_compound_grid(
            weapon_results["raw"], weapon_name_map
        )
        grid_weapon_ap = build_compound_grid(
            weapon_results["ap_weighted"], weapon_name_map
        )
        grid_weapon_family = build_compound_grid(
            weapon_results["family"], weapon_name_map
        )

        # Sub grids
        grid_sub_raw = build_compound_grid(sub_results["raw"], sub_name_map)
        grid_sub_ap = build_compound_grid(
            sub_results["ap_weighted"], sub_name_map
        )
        grid_sub_family = build_compound_grid(
            sub_results["family"], sub_name_map
        )

        # Special grids
        grid_special_raw = build_compound_grid(
            special_results["raw"], special_name_map
        )
        grid_special_ap = build_compound_grid(
            special_results["ap_weighted"], special_name_map
        )
        grid_special_family = build_compound_grid(
            special_results["family"], special_name_map
        )

        status = dbc.Alert(
            f"Processed {n_examples:,} examples across all analyses.",
            color="success",
        )

        # ========== Markdown export ==========
        markdown_data = {
            "Ability-Only Analysis": [
                (mode_labels[m], ability_results[m][1]) for m in modes
            ],
            "Weapon Analysis": [
                (mode_labels[m], weapon_results[m][1]) for m in modes
            ],
            "Sub Weapon Analysis (rollup)": [
                (mode_labels[m], sub_results[m][1]) for m in modes
            ],
            "Special Analysis (rollup)": [
                (mode_labels[m], special_results[m][1]) for m in modes
            ],
        }
        category_mappings = {
            "weapon": weapon_name_map,
            "sub": sub_name_map,
            "special": special_name_map,
        }
        markdown = format_pagerank_markdown(
            feature_id,
            feature_name,
            n_examples,
            markdown_data,
            category_mappings,
            truncate=bool(truncate_mode),
        )

        return (
            status,
            # Ability charts
            fig_ability_raw,
            fig_ability_ap,
            fig_ability_family,
            # Weapon charts
            fig_weapon_raw,
            fig_weapon_ap,
            fig_weapon_family,
            # Sub charts
            fig_sub_raw,
            fig_sub_ap,
            fig_sub_family,
            # Special charts
            fig_special_raw,
            fig_special_ap,
            fig_special_family,
            # Ability grids
            grid_ability_raw,
            grid_ability_ap,
            grid_ability_family,
            # Weapon grids
            grid_weapon_raw,
            grid_weapon_ap,
            grid_weapon_family,
            # Sub grids
            grid_sub_raw,
            grid_sub_ap,
            grid_sub_family,
            # Special grids
            grid_special_raw,
            grid_special_ap,
            grid_special_family,
            # Error and markdown
            "",
            markdown,
        )

    except Exception as e:
        logger.error(f"PageRank error: {e}", exc_info=True)
        return tuple(
            [dbc.Alert(f"Error running PageRank: {str(e)}", color="danger")]
            + [empty_fig] * 12
            + [[]] * 12
            + [str(e), ""]
        )
