import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import torch
from dash import ALL, Input, Output, State, callback_context, dcc, html

from splatnlp.dashboard.utils.converters import (
    generate_weapon_name_mapping,
    get_weapon_properties_df,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation Data Cache
# ---------------------------------------------------------------------------

# Simple cache for activation data to avoid repeated loading
# Key: feature_id, Value: (data, timestamp)
_activation_cache: dict = {}
_CACHE_MAX_SIZE = 5  # Keep last 5 features in cache


def get_cached_activations(feature_id: int):
    """Get activation data from cache or load it."""
    import time

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Check cache
    if feature_id in _activation_cache:
        data, ts = _activation_cache[feature_id]
        logger.debug(f"[Cache] Hit for feature {feature_id}")
        return data

    # Load from database
    logger.info(f"[Cache] Miss for feature {feature_id}, loading...")
    db = DASHBOARD_CONTEXT.db
    try:
        data = db.get_all_feature_activations_for_pagerank(feature_id)
    except Exception as e:
        logger.error(f"Failed to get activations: {e}")
        return None

    if data is not None and len(data) > 0:
        # Evict oldest if cache full
        if len(_activation_cache) >= _CACHE_MAX_SIZE:
            oldest_key = min(
                _activation_cache.keys(), key=lambda k: _activation_cache[k][1]
            )
            del _activation_cache[oldest_key]
            logger.debug(f"[Cache] Evicted feature {oldest_key}")

        _activation_cache[feature_id] = (data, time.time())
        logger.info(
            f"[Cache] Stored feature {feature_id} ({len(data)} examples)"
        )

    return data


def clear_activation_cache():
    """Clear the activation cache."""
    global _activation_cache
    _activation_cache = {}
    logger.info("[Cache] Cleared")


# ---------------------------------------------------------------------------
# Data Structures for Token Sensitivity Sweep
# ---------------------------------------------------------------------------


@dataclass
class TokenSweepResult:
    """Result of testing a single token."""

    token_name: str
    add_delta: float | None  # Activation change when adding token
    remove_delta: (
        float | None
    )  # Activation change when removing (None if not in base)
    was_in_base: bool  # Whether token was in base build


@dataclass
class WeaponSweepResult:
    """Result of testing a single weapon swap."""

    weapon_name: str  # Display name
    weapon_id: int  # Token ID
    swap_delta: float | None  # Activation change when swapping to this weapon
    is_current: bool  # Whether this is the current weapon


@dataclass
class SweepResults:
    """Full sweep results."""

    feature_id: int
    feature_name: str
    base_activation: float
    base_build_tokens: list[str]
    weapon_name: str
    token_results: list[TokenSweepResult]
    weapon_results: list[WeaponSweepResult]


# Regex to parse ability tokens: "swim_speed_up_21" -> ("swim_speed_up", 21)
ABILITY_FAMILY_RE = re.compile(r"^([a-z_]+?)_(\d+)$")


def parse_token(token: str) -> tuple[str, int | None]:
    """
    Parse token into (family_name, ap_value).

    Examples:
        "swim_speed_up_21" -> ("swim_speed_up", 21)
        "ninja_squid" -> ("ninja_squid", None)
    """
    match = ABILITY_FAMILY_RE.match(token)
    if match:
        return match.group(1), int(match.group(2))
    return token, None


@dataclass
class AbilityFamilyResult:
    """Results for one ability family across AP values."""

    family_name: str  # e.g., "swim_speed_up"
    ap_values: list[int] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)


@dataclass
class MainOnlyResult:
    """Result for a main-only ability."""

    ability_name: str
    delta: float


@dataclass
class WeaponGroupResult:
    """Results for weapons in a group."""

    group_name: str  # e.g., "Ink Jet" (special) or "Shooter" (class)
    weapon_names: list[str] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)
    avg_delta: float = 0.0


@dataclass
class FullSweepResults:
    """Complete results for a full sweep of one build."""

    build_id: int  # Index in queue
    build_description: str  # Human-readable build description
    feature_id: int
    feature_name: str
    base_activation: float

    # Token results
    ability_families: list[AbilityFamilyResult] = field(default_factory=list)
    main_only_abilities: list[MainOnlyResult] = field(default_factory=list)

    # Weapon results (with base build abilities)
    by_special: list[WeaponGroupResult] = field(default_factory=list)
    by_sub: list[WeaponGroupResult] = field(default_factory=list)
    by_class: list[WeaponGroupResult] = field(default_factory=list)

    # Weapon results with NULL base (no abilities - pure weapon effect)
    null_base_activation: float = 0.0
    null_base_by_special: list[WeaponGroupResult] = field(default_factory=list)
    null_base_by_sub: list[WeaponGroupResult] = field(default_factory=list)
    null_base_by_class: list[WeaponGroupResult] = field(default_factory=list)
    # Individual weapon activations (filtered by threshold)
    # Each entry: (name, activation, special, sub, weapon_class)
    null_base_weapons: list[tuple[str, float, str, str, str]] = field(
        default_factory=list
    )


@dataclass
class BuildQueueItem:
    """A build in the queue."""

    weapon_id: int
    weapon_name: str
    ability_tokens: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FrequentItemset:
    """Result of frequent itemset mining."""

    tokens: tuple[str, ...]  # The token combination
    support_high: float  # Support in high-activation examples
    support_baseline: float  # Support in baseline examples
    lift: float  # Ratio of support_high / support_baseline
    count_high: int  # Raw count in high activation examples


@dataclass
class ContrastiveItemsetResults:
    """Results of contrastive frequent itemset analysis."""

    feature_id: int
    high_threshold_pct: float  # e.g., 1.0 for top 1%
    n_high_examples: int
    n_baseline_examples: int
    itemsets_by_size: dict  # size (2,3,4) -> list[FrequentItemset]


@dataclass
class InteractionScore:
    """Pairwise or 3-way interaction score."""

    tokens: tuple[str, ...]  # 2 or 3 tokens
    interaction_value: float  # I > 0: synergy, I < 0: redundancy
    std_error: float  # Standard error across base contexts
    n_contexts: int  # Number of base contexts tested


@dataclass
class InteractionSweepResults:
    """Results of interaction sweep analysis."""

    feature_id: int
    family_mode: bool  # Whether tokens were collapsed by family
    candidate_tokens: list[str]  # The pool of tokens tested
    n_base_contexts: int  # Number of contexts sampled
    pairwise: list[InteractionScore]
    three_way: list[InteractionScore] | None  # Optional


@dataclass
class MinimalActivatingSet:
    """A minimal subset of tokens that maintains activation."""

    original_tokens: list[str]  # Full original token set
    minimal_tokens: list[str]  # Minimal subset found
    weapon_name: str  # Weapon used for this example
    original_activation: float
    minimal_activation: float
    retention_ratio: float  # minimal_activation / original_activation


@dataclass
class MinimalSetResults:
    """Results of minimal activating sets analysis."""

    feature_id: int
    threshold_ratio: float  # e.g., 0.8 for 80% retention
    n_examples_analyzed: int
    minimal_sets: list[MinimalActivatingSet]
    common_cores: list[tuple[tuple[str, ...], int]]  # (core_tokens, frequency)


# Define the layout for the Ablation tab
layout = html.Div(
    [
        dcc.Store(id="ablation-primary-store"),  # Store for primary build data
        dcc.Store(
            id="ablation-secondary-store"
        ),  # Store for modified/secondary build data
        dcc.Store(
            id="ablation-custom-build-store"
        ),  # Store for custom build data
        html.H3("Ablation Analysis"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Primary Build"),
                        # Toggle between Top Examples and Custom Build
                        dbc.RadioItems(
                            id="primary-build-source",
                            options=[
                                {
                                    "label": "From Top Examples",
                                    "value": "examples",
                                },
                                {"label": "Custom Build", "value": "custom"},
                            ],
                            value="examples",
                            inline=True,
                            className="mb-2",
                        ),
                        # Display for Top Examples primary build
                        html.Div(
                            id="primary-build-display",
                            children="No primary build selected.",
                            className="mb-3 p-2 border rounded bg-light",
                            style={"minHeight": "100px"},
                        ),
                        # Custom build input (shown when Custom Build is selected)
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Label(
                                            "Select Weapon:",
                                            className="fw-bold",
                                        ),
                                        dcc.Dropdown(
                                            id="custom-primary-weapon-dropdown",
                                            options=[],
                                            placeholder="Select weapon...",
                                            className="mb-2",
                                        ),
                                        html.Label(
                                            "Select Abilities:",
                                            className="fw-bold",
                                        ),
                                        dcc.Dropdown(
                                            id="custom-primary-abilities-dropdown",
                                            options=[],
                                            multi=True,
                                            placeholder="Select abilities...",
                                            className="mb-2",
                                        ),
                                        html.Div(
                                            id="custom-primary-summary",
                                            className="text-muted small",
                                        ),
                                    ]
                                ),
                                className="mb-3",
                            ),
                            id="custom-build-collapse",
                            is_open=False,
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.H4("Secondary Build Input (Modify Build)"),
                        html.Label("Select Weapon:"),
                        dcc.Dropdown(
                            id="secondary-weapon-dropdown",
                            options=[],
                            placeholder="Select weapon (or keep original)...",
                            className="mb-2",
                        ),
                        html.Label("Select Abilities:"),
                        dcc.Dropdown(
                            id="secondary-build-input",
                            options=[],
                            multi=True,
                            placeholder="Select abilities...",
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Run Ablation Analysis",
                            id="run-ablation-button",
                            color="success",
                            className="mt-2",
                        ),
                    ],
                    md=6,
                ),
            ]
        ),
        html.Hr(),
        html.H4("Ablation Results"),
        html.Div(
            id="ablation-results-display",
            children="Ablation results will appear here.",  # Placeholder
            className="mt-3 p-2 border rounded",
        ),
        html.Hr(className="my-4"),
        # Token Sensitivity Sweep Section
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        html.P(
                            "Test how adding/removing tokens or swapping weapons affects "
                            "the selected feature's activation.",
                            className="text-muted small mb-3",
                        ),
                        # Base build display
                        html.Div(
                            id="sweep-base-build-display",
                            className="mb-3 p-2 border rounded bg-light",
                        ),
                        # Token selection
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Tokens to Test:",
                                            className="fw-bold",
                                        ),
                                        dcc.Dropdown(
                                            id="sweep-tokens-dropdown",
                                            options=[],
                                            multi=True,
                                            placeholder="Select tokens to test...",
                                            className="mb-2",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Weapons to Test:",
                                            className="fw-bold",
                                        ),
                                        dcc.Dropdown(
                                            id="sweep-weapons-dropdown",
                                            options=[],
                                            multi=True,
                                            placeholder="Select weapons to test...",
                                            className="mb-2",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dbc.Button(
                                            "Run Sweep",
                                            id="run-sweep-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                    ],
                                    md=2,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Results display
                        dcc.Loading(
                            id="loading-sweep-results",
                            type="default",
                            children=html.Div(id="sweep-results-display"),
                        ),
                    ],
                    title="Sensitivity Sweep",
                    item_id="sweep-accordion",
                ),
            ],
            id="sweep-accordion-wrapper",
            start_collapsed=True,
            className="mt-3",
        ),
        # Full Sweep Analysis Section
        dcc.Store(id="full-sweep-build-queue", data=[]),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        html.P(
                            "Run a complete sweep of ALL tokens and ALL weapons "
                            "to analyze their effect on the selected feature.",
                            className="text-muted small mb-3",
                        ),
                        # Build Queue Display
                        html.Label("Build Queue:", className="fw-bold"),
                        html.Div(
                            id="full-sweep-queue-display",
                            className="mb-3 p-2 border rounded bg-light",
                            style={"minHeight": "60px"},
                        ),
                        # Token Statistics Display (shows families/AP present in builds)
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(
                                            "Base Build Token Statistics",
                                            className="mb-2",
                                        ),
                                        html.Div(
                                            id="build-token-statistics-display",
                                            className="small",
                                        ),
                                    ]
                                ),
                                className="mb-3",
                            ),
                            id="build-token-statistics-collapse",
                            is_open=False,
                        ),
                        # Queue management buttons
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Add Primary Build",
                                        id="add-primary-to-queue-button",
                                        color="secondary",
                                        size="sm",
                                        className="me-2",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Clear Queue",
                                        id="clear-queue-button",
                                        color="outline-danger",
                                        size="sm",
                                        className="me-2",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Run Full Sweep",
                                        id="run-full-sweep-button",
                                        color="success",
                                        size="sm",
                                    ),
                                    width="auto",
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Aggregation toggle
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Aggregation:",
                                            className="fw-bold me-2",
                                        ),
                                        dbc.RadioItems(
                                            id="full-sweep-aggregation-method",
                                            options=[
                                                {
                                                    "label": "Mean",
                                                    "value": "mean",
                                                },
                                                {
                                                    "label": "Median",
                                                    "value": "median",
                                                },
                                            ],
                                            value="mean",
                                            inline=True,
                                        ),
                                    ],
                                    width="auto",
                                    className="d-flex align-items-center",
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Progress display
                        html.Div(
                            id="full-sweep-progress",
                            className="mb-3 text-muted small",
                        ),
                        # Results display
                        dcc.Loading(
                            id="loading-full-sweep-results",
                            type="default",
                            children=html.Div(id="full-sweep-results-display"),
                        ),
                    ],
                    title="Full Sweep Analysis",
                    item_id="full-sweep-accordion",
                ),
            ],
            id="full-sweep-accordion-wrapper",
            start_collapsed=True,
            className="mt-3",
        ),
        # ---------------------------------------------------------------------------
        # Higher-Order Interaction Analysis Section
        # ---------------------------------------------------------------------------
        html.Hr(className="my-4"),
        html.H4("Higher-Order Interaction Analysis"),
        html.P(
            "Discover token combinations, synergies, and minimal activating patterns.",
            className="text-muted small mb-3",
        ),
        dbc.Accordion(
            [
                # 1. Contrastive Frequent Itemsets
                dbc.AccordionItem(
                    [
                        html.P(
                            "Mine token combinations that are over-represented in "
                            "high-activation examples compared to baseline.",
                            className="text-muted small mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "High Activation Threshold:",
                                            className="fw-bold",
                                        ),
                                        dcc.Dropdown(
                                            id="itemset-high-threshold",
                                            options=[
                                                {
                                                    "label": "Top 1%",
                                                    "value": 1.0,
                                                },
                                                {
                                                    "label": "Top 5%",
                                                    "value": 5.0,
                                                },
                                                {
                                                    "label": "Top 10%",
                                                    "value": 10.0,
                                                },
                                            ],
                                            value=5.0,
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Mode:", className="fw-bold"
                                        ),
                                        dbc.Checklist(
                                            id="itemset-family-mode",
                                            options=[
                                                {
                                                    "label": "Collapse by family",
                                                    "value": "family",
                                                }
                                            ],
                                            value=[],
                                            inline=True,
                                            className="mt-2",
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dbc.Button(
                                            "Run Itemset Mining",
                                            id="run-itemset-mining-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                    ],
                                    md=2,
                                    className="d-flex align-items-end",
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dcc.Clipboard(
                                            target_id="itemset-markdown-content",
                                            title="Copy results as Markdown",
                                            className="btn btn-outline-secondary w-100",
                                            style={"fontSize": "0.9rem"},
                                        ),
                                    ],
                                    md=1,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Hidden div for markdown clipboard content
                        html.Div(
                            "",
                            id="itemset-markdown-content",
                            style={"display": "none"},
                        ),
                        dcc.Loading(
                            id="loading-itemset-results",
                            type="default",
                            children=html.Div(id="itemset-results-display"),
                        ),
                    ],
                    title="Contrastive Frequent Itemsets",
                    item_id="itemset-accordion",
                ),
                # 2. Interaction Sweep (Pairwise + 3-way)
                dbc.AccordionItem(
                    [
                        html.P(
                            "Compute pairwise synergy/redundancy scores between tokens. "
                            "I > 0 means synergy (combo matters), I < 0 means redundancy.",
                            className="text-muted small mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Candidate Tokens (select 10-50):",
                                            className="fw-bold",
                                        ),
                                        dcc.Dropdown(
                                            id="interaction-tokens-dropdown",
                                            options=[],
                                            multi=True,
                                            placeholder="Select tokens to test...",
                                        ),
                                    ],
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "# Base Contexts:",
                                            className="fw-bold",
                                        ),
                                        dbc.Input(
                                            id="interaction-n-contexts",
                                            type="number",
                                            value=50,
                                            min=10,
                                            max=200,
                                            step=10,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Options:", className="fw-bold"
                                        ),
                                        dbc.Checklist(
                                            id="interaction-options",
                                            options=[
                                                {
                                                    "label": "Family mode",
                                                    "value": "family",
                                                },
                                                {
                                                    "label": "Include 3-way",
                                                    "value": "three_way",
                                                },
                                            ],
                                            value=[],
                                            inline=True,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dbc.Button(
                                            "Run Sweep",
                                            id="run-interaction-sweep-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                    ],
                                    md=1,
                                    className="d-flex align-items-end",
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dcc.Clipboard(
                                            target_id="interaction-markdown-content",
                                            title="Copy results as Markdown",
                                            className="btn btn-outline-secondary w-100",
                                            style={"fontSize": "0.9rem"},
                                        ),
                                    ],
                                    md=1,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Hidden div for markdown clipboard content
                        html.Div(
                            "",
                            id="interaction-markdown-content",
                            style={"display": "none"},
                        ),
                        dcc.Loading(
                            id="loading-interaction-results",
                            type="default",
                            children=html.Div(id="interaction-results-display"),
                        ),
                    ],
                    title="Interaction Sweep",
                    item_id="interaction-accordion",
                ),
                # 3. Minimal Activating Sets
                dbc.AccordionItem(
                    [
                        html.P(
                            "Find the smallest token subsets that maintain feature activation. "
                            "Useful for identifying core patterns and redundant tokens.",
                            className="text-muted small mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Retention Threshold:",
                                            className="fw-bold",
                                        ),
                                        dcc.Slider(
                                            id="minimal-set-threshold",
                                            min=0.5,
                                            max=0.95,
                                            step=0.05,
                                            value=0.8,
                                            marks={
                                                0.5: "50%",
                                                0.6: "60%",
                                                0.7: "70%",
                                                0.8: "80%",
                                                0.9: "90%",
                                            },
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(
                                            "# Examples to Analyze:",
                                            className="fw-bold",
                                        ),
                                        dbc.Input(
                                            id="minimal-set-n-examples",
                                            type="number",
                                            value=50,
                                            min=10,
                                            max=200,
                                            step=10,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dbc.Button(
                                            "Find Minimal Sets",
                                            id="run-minimal-sets-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                    ],
                                    md=2,
                                    className="d-flex align-items-end",
                                ),
                                dbc.Col(
                                    [
                                        html.Label(" ", className="d-block"),
                                        dcc.Clipboard(
                                            target_id="minimal-sets-markdown-content",
                                            title="Copy results as Markdown",
                                            className="btn btn-outline-secondary w-100",
                                            style={"fontSize": "0.9rem"},
                                        ),
                                    ],
                                    md=1,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Hidden div for markdown clipboard content
                        html.Div(
                            "",
                            id="minimal-sets-markdown-content",
                            style={"display": "none"},
                        ),
                        dcc.Loading(
                            id="loading-minimal-sets-results",
                            type="default",
                            children=html.Div(
                                id="minimal-sets-results-display"
                            ),
                        ),
                    ],
                    title="Minimal Activating Sets",
                    item_id="minimal-sets-accordion",
                ),
            ],
            id="higher-order-accordion-wrapper",
            start_collapsed=True,
            className="mt-3",
        ),
    ]
)


# Callback to populate ability dropdown options
@dash.callback(
    Output("secondary-build-input", "options"),
    Input("page-load-trigger", "data"),
)
def populate_ability_dropdown(_):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "vocab")
        or DASHBOARD_CONTEXT.vocab is None
    ):
        return []

    # Get all ability tokens (exclude special tokens that start with <)
    return [
        {"label": tok, "value": tok}
        for tok in sorted(DASHBOARD_CONTEXT.vocab.keys())
        if not tok.startswith("<")
    ]


# Callback to populate weapon dropdown with English names
@dash.callback(
    Output("secondary-weapon-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_weapon_dropdown(_):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab")
        or DASHBOARD_CONTEXT.inv_weapon_vocab is None
    ):
        return []

    # Generate weapon name mapping to get English names
    weapon_name_mapping = generate_weapon_name_mapping(
        DASHBOARD_CONTEXT.inv_weapon_vocab
    )

    # Create options with English names as labels and weapon IDs as values
    options = []
    for weapon_id, raw_name in DASHBOARD_CONTEXT.inv_weapon_vocab.items():
        english_name = weapon_name_mapping.get(weapon_id, raw_name)
        options.append(
            {
                "label": english_name,
                "value": weapon_id,  # Store the ID as the value
            }
        )

    # Sort by English name
    options.sort(key=lambda x: x["label"])

    return options


# Callback to populate custom primary weapon dropdown
@dash.callback(
    Output("custom-primary-weapon-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_custom_weapon_dropdown(_):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab")
        or DASHBOARD_CONTEXT.inv_weapon_vocab is None
    ):
        return []

    weapon_name_mapping = generate_weapon_name_mapping(
        DASHBOARD_CONTEXT.inv_weapon_vocab
    )

    options = []
    for weapon_id, raw_name in DASHBOARD_CONTEXT.inv_weapon_vocab.items():
        english_name = weapon_name_mapping.get(weapon_id, raw_name)
        options.append({"label": english_name, "value": weapon_id})

    options.sort(key=lambda x: x["label"])
    return options


# Callback to populate custom primary abilities dropdown
@dash.callback(
    Output("custom-primary-abilities-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_custom_abilities_dropdown(_):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "vocab")
        or DASHBOARD_CONTEXT.vocab is None
    ):
        return []

    return [
        {"label": tok, "value": tok}
        for tok in sorted(DASHBOARD_CONTEXT.vocab.keys())
        if not tok.startswith("<")
    ]


# Callback to toggle custom build collapse
@dash.callback(
    Output("custom-build-collapse", "is_open"),
    Input("primary-build-source", "value"),
)
def toggle_custom_build_collapse(source):
    return source == "custom"


# Callback to show custom build summary
@dash.callback(
    Output("custom-primary-summary", "children"),
    [
        Input("custom-primary-weapon-dropdown", "value"),
        Input("custom-primary-abilities-dropdown", "value"),
    ],
)
def update_custom_build_summary(weapon_id, abilities):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not weapon_id and not abilities:
        return "Select a weapon and abilities to create a custom build."

    parts = []

    if weapon_id:
        inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
        weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
        weapon_name = weapon_name_mapping.get(weapon_id, str(weapon_id))
        parts.append(f"Weapon: {weapon_name}")

    if abilities:
        parts.append(f"Abilities: {len(abilities)} selected")

    return " | ".join(parts)


# Callback to display primary build details and pre-fill secondary input
@dash.callback(
    [
        Output("primary-build-display", "children"),
        Output("secondary-build-input", "value"),
        Output("secondary-weapon-dropdown", "value"),
    ],
    Input("ablation-primary-store", "data"),
)
def display_primary_build(primary_data):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT  # Moved import here

    if primary_data:
        inv_vocab = (
            DASHBOARD_CONTEXT.inv_vocab
            if hasattr(DASHBOARD_CONTEXT, "inv_vocab")
            else {}
        )
        inv_weapon_vocab = (
            DASHBOARD_CONTEXT.inv_weapon_vocab
            if hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab")
            else {}
        )
        weapon_id = primary_data.get("weapon_id_token", "N/A")
        ability_token_ids = primary_data.get("ability_input_tokens", [])
        activation = primary_data.get("activation", "N/A")

        ability_names = [
            str(inv_vocab.get(token_id, token_id))
            for token_id in ability_token_ids
        ]
        abilities_display_str = (
            ", ".join(ability_names) if ability_names else "None"
        )

        # Get English weapon name from ID
        weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
        weapon_raw_name = inv_weapon_vocab.get(weapon_id, f"Weapon {weapon_id}")
        weapon_english_name = weapon_name_mapping.get(
            weapon_id, weapon_raw_name
        )

        display_children = [
            html.P(f"Weapon: {weapon_english_name}"),
            html.P(f"Abilities: {abilities_display_str}"),
            html.P(
                f"Original Activation: {activation:.4f}"
                if isinstance(activation, float)
                else f"Original Activation: {activation}"
            ),
        ]
        return (
            display_children,
            ability_names,  # Pre-fill abilities
            weapon_id,  # Pre-fill weapon with same weapon
        )
    else:
        return "No primary build selected.", [], None


# Callback to update the secondary store from the dropdown
@dash.callback(
    Output("ablation-secondary-store", "data"),
    Input("secondary-build-input", "value"),
)
def update_secondary_store(secondary_build_list):
    # Store the list of ability names
    if secondary_build_list is None:
        return dash.no_update
    return {"ability_tokens_list": secondary_build_list}


# Helper function to compute feature activation
@torch.no_grad()
def compute_feature_activation(
    ability_names: list[str], weapon_name: str, feature_id: int
):
    """Compute SAE feature activation for a given build and weapon."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Debug logging (comment out for production)
    # print(f"DEBUG: compute_feature_activation called with:")
    # print(f"  ability_names: {ability_names}")
    # print(f"  weapon_name: {weapon_name}")
    # print(f"  feature_id: {feature_id}")

    if feature_id is None:
        print("ERROR: feature_id is None")
        return None

    if not hasattr(DASHBOARD_CONTEXT, "primary_model"):
        print("ERROR: DASHBOARD_CONTEXT has no primary_model")
        return None

    if DASHBOARD_CONTEXT.primary_model is None:
        print("ERROR: primary_model is None")
        return None

    if not hasattr(DASHBOARD_CONTEXT, "sae_model"):
        print("ERROR: DASHBOARD_CONTEXT has no sae_model")
        return None

    if DASHBOARD_CONTEXT.sae_model is None:
        print("ERROR: sae_model is None")
        return None

    vocab = DASHBOARD_CONTEXT.vocab
    weapon_vocab = DASHBOARD_CONTEXT.weapon_vocab
    device = getattr(DASHBOARD_CONTEXT, "device", "cpu")
    pad_id = vocab.get("<PAD>", 0)

    # Convert ability names to token IDs
    token_ids = [vocab.get(tok, pad_id) for tok in ability_names]

    # Check weapon lookup
    if weapon_name not in weapon_vocab:
        print(f"ERROR: Weapon '{weapon_name}' not in weapon_vocab")
        print(f"  Available weapons (first 5): {list(weapon_vocab.keys())[:5]}")
        return None

    weapon_id = weapon_vocab.get(weapon_name, 0)

    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)
    weapon_token = torch.tensor([weapon_id], device=device).unsqueeze(0)
    mask = tokens == pad_id

    model = DASHBOARD_CONTEXT.primary_model
    sae = DASHBOARD_CONTEXT.sae_model

    try:
        ability_embeddings = model.ability_embedding(tokens)
        weapon_embeddings = model.weapon_embedding(weapon_token).expand_as(
            ability_embeddings
        )
        embeddings = ability_embeddings + weapon_embeddings
        x = model.input_proj(embeddings)

        for layer in model.transformer_layers:
            x = layer(x, key_padding_mask=mask)

        masked = model.masked_mean(x, mask)
        _, hidden = sae.encode(masked)

        result = hidden[0, feature_id].item()
        return result

    except Exception as e:
        print(f"ERROR during compute_feature_activation: {e}")
        import traceback

        traceback.print_exc()
        return None


@dash.callback(
    Output("ablation-results-display", "children"),
    Input("run-ablation-button", "n_clicks"),
    [
        State("ablation-primary-store", "data"),
        State("secondary-build-input", "value"),
        State("secondary-weapon-dropdown", "value"),
        State("feature-dropdown", "value"),
        State("primary-build-source", "value"),
        State("custom-primary-weapon-dropdown", "value"),
        State("custom-primary-abilities-dropdown", "value"),
    ],
)
def run_ablation_analysis(
    n_clicks,
    primary_data,
    secondary_build_list,
    secondary_weapon_id,
    selected_feature_id,
    build_source,
    custom_weapon_id,
    custom_abilities,
):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return html.P(
            "Enter primary and secondary build details, then click 'Run Ablation Analysis'. Select a feature from the dropdown on the left to see its specific ablation."
        )

    if selected_feature_id is None or selected_feature_id == -1:
        return html.P(
            "Please select a feature from the dropdown on the left to see its specific ablation analysis."
        )

    # Get vocab mappings
    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    # Determine primary build based on source
    if build_source == "custom":
        # Use custom build
        if not custom_weapon_id:
            return dbc.Alert(
                "Please select a weapon for the custom build.", color="warning"
            )

        primary_weapon_id = custom_weapon_id
        primary_ability_names = custom_abilities if custom_abilities else []
    else:
        # Use primary from Top Examples
        if not primary_data:
            return dbc.Alert(
                "Please select a primary build from Top Examples.",
                color="warning",
            )

        primary_weapon_id = primary_data.get("weapon_id_token")
        primary_ability_token_ids = primary_data.get("ability_input_tokens", [])

        if primary_weapon_id is None:
            return dbc.Alert(
                "Weapon ID token missing from primary build data.",
                color="danger",
            )

        # Convert primary token IDs back to names
        primary_ability_names = [
            inv_vocab.get(token_id, f"Unknown_{token_id}")
            for token_id in primary_ability_token_ids
        ]

    # Secondary build is optional - if not provided, just show primary activation
    has_secondary = secondary_build_list and len(secondary_build_list) > 0

    # Get weapon names (use secondary weapon if selected, otherwise use primary)
    if secondary_weapon_id is None:
        secondary_weapon_id = primary_weapon_id

    primary_weapon_raw = inv_weapon_vocab.get(
        primary_weapon_id, f"Unknown_weapon_{primary_weapon_id}"
    )
    secondary_weapon_raw = inv_weapon_vocab.get(
        secondary_weapon_id, f"Unknown_weapon_{secondary_weapon_id}"
    )

    # Compute primary activation
    primary_activation = compute_feature_activation(
        primary_ability_names, primary_weapon_raw, selected_feature_id
    )
    if primary_activation is None:
        return dbc.Alert(
            "Could not compute primary activation. Check model and inputs.",
            color="danger",
        )

    # Get feature display name
    feature_name_or_id = f"Feature {selected_feature_id}"
    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        feature_name_or_id = (
            DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
                selected_feature_id
            )
        )

    # Get English weapon names for display
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
    primary_weapon_english = weapon_name_mapping.get(
        primary_weapon_id, primary_weapon_raw
    )

    # Build result display
    primary_abilities_str = (
        ", ".join(primary_ability_names) if primary_ability_names else "(empty)"
    )

    result_sections = [
        html.H5(f"Ablation for {feature_name_or_id}:"),
        html.P(
            f"Primary Build: {primary_abilities_str} + {primary_weapon_english}"
        ),
        html.P(f"Primary Activation: {primary_activation:.4f}"),
    ]

    # If secondary build provided, compute and show comparison
    if has_secondary:
        secondary_activation = compute_feature_activation(
            secondary_build_list, secondary_weapon_raw, selected_feature_id
        )

        if secondary_activation is None:
            return dbc.Alert(
                "Could not compute secondary activation. Check model and inputs.",
                color="danger",
            )

        diff = secondary_activation - primary_activation
        secondary_weapon_english = weapon_name_mapping.get(
            secondary_weapon_id, secondary_weapon_raw
        )

        result_sections.extend(
            [
                html.Hr(),
                html.P(
                    f"Secondary Build: {', '.join(secondary_build_list)} + {secondary_weapon_english}"
                ),
                html.P(f"Secondary Activation: {secondary_activation:.4f}"),
                html.Hr(),
                html.P(
                    f"Difference: {diff:.4f}",
                    style={
                        "font-weight": "bold",
                        "color": "green" if diff > 0 else "red",
                    },
                ),
            ]
        )

        if primary_weapon_id != secondary_weapon_id:
            result_sections.append(
                html.P(
                    "⚠️ Weapon changed",
                    style={"font-style": "italic", "color": "blue"},
                )
            )
    else:
        # No secondary - just show primary result
        result_sections.append(
            html.P(
                "No secondary build specified. Add abilities to the secondary build to compare.",
                className="text-muted mt-3",
            )
        )

    return html.Div(result_sections)


# ---------------------------------------------------------------------------
# Token Sensitivity Sweep Callbacks and Functions
# ---------------------------------------------------------------------------


@dash.callback(
    Output("sweep-tokens-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_sweep_tokens_dropdown(_):
    """Populate the sweep tokens dropdown with all ability tokens."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "vocab")
        or DASHBOARD_CONTEXT.vocab is None
    ):
        return []

    # Get all ability tokens (exclude special tokens that start with <)
    return [
        {"label": tok, "value": tok}
        for tok in sorted(DASHBOARD_CONTEXT.vocab.keys())
        if not tok.startswith("<")
    ]


@dash.callback(
    Output("sweep-weapons-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_sweep_weapons_dropdown(_):
    """Populate the sweep weapons dropdown with all weapons."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab")
        or DASHBOARD_CONTEXT.inv_weapon_vocab is None
    ):
        return []

    # Generate weapon name mapping to get English names
    weapon_name_mapping = generate_weapon_name_mapping(
        DASHBOARD_CONTEXT.inv_weapon_vocab
    )

    # Create options with English names as labels and weapon IDs as values
    options = []
    for weapon_id, raw_name in DASHBOARD_CONTEXT.inv_weapon_vocab.items():
        english_name = weapon_name_mapping.get(weapon_id, raw_name)
        options.append(
            {
                "label": english_name,
                "value": weapon_id,  # Store the token ID as the value
            }
        )

    # Sort by English name
    options.sort(key=lambda x: x["label"])

    return options


@dash.callback(
    Output("sweep-base-build-display", "children"),
    [
        Input("ablation-primary-store", "data"),
        Input("feature-dropdown", "value"),
        Input("primary-build-source", "value"),
        Input("custom-primary-weapon-dropdown", "value"),
        Input("custom-primary-abilities-dropdown", "value"),
    ],
)
def display_sweep_base_build(
    primary_data, feature_id, build_source, custom_weapon_id, custom_abilities
):
    """Display the base build info in the sweep section."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)

    # Determine build source
    if build_source == "custom":
        if not custom_weapon_id:
            return html.P(
                "Select a weapon for your custom build.",
                className="text-muted mb-0",
            )
        weapon_id = custom_weapon_id
        ability_names = custom_abilities if custom_abilities else []
    else:
        if not primary_data:
            return html.P(
                "Select a primary build from Top Examples to use as the base build.",
                className="text-muted mb-0",
            )
        weapon_id = primary_data.get("weapon_id_token", "N/A")
        ability_token_ids = primary_data.get("ability_input_tokens", [])
        ability_names = [
            str(inv_vocab.get(token_id, token_id))
            for token_id in ability_token_ids
        ]

    abilities_str = ", ".join(ability_names) if ability_names else "(empty)"

    # Get weapon name
    weapon_raw = inv_weapon_vocab.get(weapon_id, f"Weapon {weapon_id}")
    weapon_english = weapon_name_mapping.get(weapon_id, weapon_raw)

    # Get feature name
    feature_name = (
        f"Feature {feature_id}" if feature_id else "No feature selected"
    )
    if (
        feature_id
        and hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        feature_name = (
            DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
                feature_id
            )
        )

    return html.Div(
        [
            html.Strong("Base Build: ", className="me-2"),
            html.Span(f"{abilities_str} + {weapon_english}"),
            html.Br(),
            html.Strong("Testing Feature: ", className="me-2"),
            html.Span(feature_name),
        ]
    )


def compute_sensitivity_sweep(
    base_tokens: list[str],
    weapon_raw_name: str,
    base_weapon_id: int,
    tokens_to_test: list[str],
    weapons_to_test: list[int],
    feature_id: int,
) -> SweepResults | None:
    """
    Run sensitivity sweep for all queued tokens and weapons.

    For each token in tokens_to_test:
    1. Add test: add token to base, compute new activation, get delta
    2. Remove test: if token in base, remove it, compute activation, get delta

    For each weapon in weapons_to_test:
    - Swap test: replace current weapon, compute new activation, get delta

    Returns results sorted by absolute impact.
    """
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    # Compute base activation once
    base_activation = compute_feature_activation(
        base_tokens, weapon_raw_name, feature_id
    )
    if base_activation is None:
        return None

    # Get feature name
    feature_name = f"Feature {feature_id}"
    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        feature_name = (
            DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
                feature_id
            )
        )

    # Get weapon mappings
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
    weapon_display = weapon_name_mapping.get(base_weapon_id, weapon_raw_name)

    # --- Token Sweep ---
    token_results = []
    base_token_set = set(base_tokens)

    for token in tokens_to_test:
        was_in_base = token in base_token_set

        # Add test: add token to base (even if already present)
        add_tokens = base_tokens + [token]
        add_activation = compute_feature_activation(
            add_tokens, weapon_raw_name, feature_id
        )
        add_delta = (
            (add_activation - base_activation)
            if add_activation is not None
            else None
        )

        # Remove test: only if token was in base
        remove_delta = None
        if was_in_base:
            remove_tokens = [t for t in base_tokens if t != token]
            if remove_tokens:  # Don't test empty build
                remove_activation = compute_feature_activation(
                    remove_tokens, weapon_raw_name, feature_id
                )
                if remove_activation is not None:
                    remove_delta = remove_activation - base_activation

        token_results.append(
            TokenSweepResult(
                token_name=token,
                add_delta=add_delta,
                remove_delta=remove_delta,
                was_in_base=was_in_base,
            )
        )

    # Sort tokens by absolute add_delta (most impactful first)
    token_results.sort(
        key=lambda x: abs(x.add_delta) if x.add_delta is not None else 0,
        reverse=True,
    )

    # --- Weapon Sweep ---
    weapon_results = []

    for weapon_id in weapons_to_test:
        is_current = weapon_id == base_weapon_id

        # Get raw weapon name for this weapon
        test_weapon_raw = inv_weapon_vocab.get(
            weapon_id, f"Unknown_weapon_{weapon_id}"
        )
        test_weapon_display = weapon_name_mapping.get(
            weapon_id, test_weapon_raw
        )

        if is_current:
            # Current weapon has 0 delta
            swap_delta = 0.0
        else:
            # Compute activation with swapped weapon
            swap_activation = compute_feature_activation(
                base_tokens, test_weapon_raw, feature_id
            )
            swap_delta = (
                (swap_activation - base_activation)
                if swap_activation is not None
                else None
            )

        weapon_results.append(
            WeaponSweepResult(
                weapon_name=test_weapon_display,
                weapon_id=weapon_id,
                swap_delta=swap_delta,
                is_current=is_current,
            )
        )

    # Sort weapons by absolute swap_delta (most impactful first)
    weapon_results.sort(
        key=lambda x: abs(x.swap_delta) if x.swap_delta is not None else 0,
        reverse=True,
    )

    return SweepResults(
        feature_id=feature_id,
        feature_name=feature_name,
        base_activation=base_activation,
        base_build_tokens=base_tokens,
        weapon_name=weapon_display,
        token_results=token_results,
        weapon_results=weapon_results,
    )


def format_sweep_markdown(sweep_results: SweepResults) -> str:
    """Format sweep results as Markdown for clipboard export."""
    lines = [
        f"# Sensitivity Sweep: {sweep_results.feature_name}",
        "",
        f"**Base Build:** {', '.join(sweep_results.base_build_tokens)} + {sweep_results.weapon_name}",
        f"**Base Activation:** {sweep_results.base_activation:.4f}",
        "",
    ]

    # Token results
    if sweep_results.token_results:
        lines.append("## Token Effects")
        lines.append("")
        lines.append("| Token | Add Effect | Remove Effect | Notes |")
        lines.append("|-------|------------|---------------|-------|")

        for result in sweep_results.token_results:
            add_text = (
                f"{result.add_delta:+.4f}"
                if result.add_delta is not None
                else "—"
            )
            remove_text = (
                f"{result.remove_delta:+.4f}"
                if result.remove_delta is not None
                else "—"
            )
            notes = "(in base)" if result.was_in_base else ""
            lines.append(
                f"| {result.token_name} | {add_text} | {remove_text} | {notes} |"
            )

        lines.append("")

    # Weapon results
    if sweep_results.weapon_results:
        lines.append("## Weapon Swap Effects")
        lines.append("")
        lines.append("| Weapon | Swap Effect | Notes |")
        lines.append("|--------|-------------|-------|")

        for result in sweep_results.weapon_results:
            delta_text = (
                f"{result.swap_delta:+.4f}"
                if result.swap_delta is not None
                else "—"
            )
            notes = "(current)" if result.is_current else ""
            lines.append(f"| {result.weapon_name} | {delta_text} | {notes} |")

        lines.append("")

    return "\n".join(lines)


def build_sweep_results_display(sweep_results: SweepResults) -> html.Div:
    """Build the results display for token and weapon sweeps."""
    if not sweep_results:
        return html.P("No results to display.")

    has_tokens = (
        sweep_results.token_results and len(sweep_results.token_results) > 0
    )
    has_weapons = (
        sweep_results.weapon_results and len(sweep_results.weapon_results) > 0
    )

    if not has_tokens and not has_weapons:
        return html.P("No results to display.")

    # Generate markdown for clipboard
    markdown_text = format_sweep_markdown(sweep_results)

    sections = [
        # Header with base activation and copy button
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                html.Strong("Base Activation: "),
                                f"{sweep_results.base_activation:.4f}",
                            ],
                            className="mb-0",
                        ),
                    ],
                    width=10,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Clipboard(
                                    target_id="sweep-markdown-content",
                                    title="Copy to clipboard (Markdown)",
                                    className="btn btn-outline-secondary btn-sm",
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
                    width=2,
                    className="d-flex align-items-center justify-content-center",
                ),
            ],
            className="mb-3",
        ),
        # Hidden div with markdown content for clipboard
        html.Div(
            markdown_text,
            id="sweep-markdown-content",
            style={"display": "none"},
        ),
    ]

    # --- Token Results Section ---
    if has_tokens:
        token_rows = []
        max_token_delta = (
            max(
                abs(r.add_delta)
                for r in sweep_results.token_results
                if r.add_delta is not None
            )
            or 1.0
        )

        for result in sweep_results.token_results:
            # Format add delta
            if result.add_delta is not None:
                add_color = (
                    "green"
                    if result.add_delta > 0
                    else "red" if result.add_delta < 0 else "gray"
                )
                add_text = f"{result.add_delta:+.4f}"
                bar_width = min(
                    100, int(abs(result.add_delta) / max_token_delta * 100)
                )
            else:
                add_color = "gray"
                add_text = "—"
                bar_width = 0

            # Format remove delta
            if result.remove_delta is not None:
                remove_color = (
                    "green"
                    if result.remove_delta > 0
                    else "red" if result.remove_delta < 0 else "gray"
                )
                remove_text = f"{result.remove_delta:+.4f}"
            else:
                remove_text = "—"
                remove_color = "gray"

            bar_color = (
                "bg-success"
                if result.add_delta and result.add_delta > 0
                else "bg-danger"
            )

            token_rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                result.token_name,
                                (
                                    html.Span(
                                        " (in base)",
                                        className="text-muted small",
                                    )
                                    if result.was_in_base
                                    else ""
                                ),
                            ]
                        ),
                        html.Td(
                            add_text,
                            style={"color": add_color, "fontWeight": "bold"},
                        ),
                        html.Td(
                            remove_text,
                            style={"color": remove_color},
                        ),
                        html.Td(
                            html.Div(
                                html.Div(
                                    className=f"{bar_color}",
                                    style={
                                        "width": f"{bar_width}%",
                                        "height": "100%",
                                    },
                                ),
                                className="bg-light",
                                style={
                                    "height": "12px",
                                    "width": "100px",
                                    "border": "1px solid #ccc",
                                },
                            )
                        ),
                    ]
                )
            )

        sections.append(
            html.Div(
                [
                    html.H6("Token Effects", className="mb-2"),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Token"),
                                        html.Th("Add Effect"),
                                        html.Th("Remove Effect"),
                                        html.Th("Impact"),
                                    ]
                                )
                            ),
                            html.Tbody(token_rows),
                        ],
                        bordered=True,
                        hover=True,
                        size="sm",
                        className="mb-2",
                    ),
                    html.P(
                        html.Small(
                            "Add Effect: activation change when token is added. "
                            "Remove Effect: change when removed (— if not in base).",
                            className="text-muted",
                        ),
                    ),
                ],
                className="mb-4",
            )
        )

    # --- Weapon Results Section ---
    if has_weapons:
        weapon_rows = []
        max_weapon_delta = (
            max(
                abs(r.swap_delta)
                for r in sweep_results.weapon_results
                if r.swap_delta is not None
            )
            or 1.0
        )

        for result in sweep_results.weapon_results:
            if result.swap_delta is not None:
                delta_color = (
                    "green"
                    if result.swap_delta > 0
                    else "red" if result.swap_delta < 0 else "gray"
                )
                delta_text = f"{result.swap_delta:+.4f}"
                bar_width = min(
                    100, int(abs(result.swap_delta) / max_weapon_delta * 100)
                )
            else:
                delta_color = "gray"
                delta_text = "—"
                bar_width = 0

            bar_color = (
                "bg-success"
                if result.swap_delta and result.swap_delta > 0
                else "bg-danger"
            )

            weapon_rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                result.weapon_name,
                                (
                                    html.Span(
                                        " (current)",
                                        className="text-muted small",
                                    )
                                    if result.is_current
                                    else ""
                                ),
                            ]
                        ),
                        html.Td(
                            delta_text,
                            style={"color": delta_color, "fontWeight": "bold"},
                        ),
                        html.Td(
                            html.Div(
                                html.Div(
                                    className=f"{bar_color}",
                                    style={
                                        "width": f"{bar_width}%",
                                        "height": "100%",
                                    },
                                ),
                                className="bg-light",
                                style={
                                    "height": "12px",
                                    "width": "100px",
                                    "border": "1px solid #ccc",
                                },
                            )
                        ),
                    ]
                )
            )

        sections.append(
            html.Div(
                [
                    html.H6("Weapon Swap Effects", className="mb-2"),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Weapon"),
                                        html.Th("Swap Effect"),
                                        html.Th("Impact"),
                                    ]
                                )
                            ),
                            html.Tbody(weapon_rows),
                        ],
                        bordered=True,
                        hover=True,
                        size="sm",
                        className="mb-2",
                    ),
                    html.P(
                        html.Small(
                            "Swap Effect: activation change when replacing base weapon with this weapon.",
                            className="text-muted",
                        ),
                    ),
                ],
            )
        )

    return html.Div(sections)


@dash.callback(
    Output("sweep-results-display", "children"),
    Input("run-sweep-button", "n_clicks"),
    [
        State("ablation-primary-store", "data"),
        State("sweep-tokens-dropdown", "value"),
        State("sweep-weapons-dropdown", "value"),
        State("feature-dropdown", "value"),
        State("primary-build-source", "value"),
        State("custom-primary-weapon-dropdown", "value"),
        State("custom-primary-abilities-dropdown", "value"),
    ],
)
def run_sensitivity_sweep(
    n_clicks,
    primary_data,
    tokens_to_test,
    weapons_to_test,
    feature_id,
    build_source,
    custom_weapon_id,
    custom_abilities,
):
    """Run the token and weapon sensitivity sweep."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return html.P(
            "Select tokens and/or weapons to test and click 'Run Sweep'.",
            className="text-muted",
        )

    has_tokens = tokens_to_test and len(tokens_to_test) > 0
    has_weapons = weapons_to_test and len(weapons_to_test) > 0

    if not has_tokens and not has_weapons:
        return dbc.Alert(
            "Please select at least one token or weapon to test.",
            color="warning",
        )

    if feature_id is None:
        return dbc.Alert(
            "Please select a feature from the dropdown.",
            color="warning",
        )

    # Get vocab mappings
    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    # Determine build source
    if build_source == "custom":
        if not custom_weapon_id:
            return dbc.Alert(
                "Please select a weapon for the custom build.",
                color="warning",
            )
        weapon_id = custom_weapon_id
        base_tokens = custom_abilities if custom_abilities else []
    else:
        if not primary_data:
            return dbc.Alert(
                "Please select a primary build first (from Top Examples).",
                color="warning",
            )
        weapon_id = primary_data.get("weapon_id_token")
        ability_token_ids = primary_data.get("ability_input_tokens", [])

        if weapon_id is None:
            return dbc.Alert(
                "Weapon ID missing from primary build.", color="danger"
            )

        # Convert token IDs to names
        base_tokens = [
            str(inv_vocab.get(token_id, f"Unknown_{token_id}"))
            for token_id in ability_token_ids
        ]

    # Get weapon raw name
    weapon_raw = inv_weapon_vocab.get(weapon_id, f"Unknown_weapon_{weapon_id}")

    # Run the sweep
    sweep_results = compute_sensitivity_sweep(
        base_tokens=base_tokens,
        weapon_raw_name=weapon_raw,
        base_weapon_id=weapon_id,
        tokens_to_test=tokens_to_test or [],
        weapons_to_test=weapons_to_test or [],
        feature_id=feature_id,
    )

    if sweep_results is None:
        return dbc.Alert(
            "Failed to compute sweep. Check model and inputs.",
            color="danger",
        )

    return build_sweep_results_display(sweep_results)


# ---------------------------------------------------------------------------
# Full Sweep Analysis Callbacks and Functions
# ---------------------------------------------------------------------------


@dash.callback(
    Output("full-sweep-build-queue", "data", allow_duplicate=True),
    [
        Input("add-primary-to-queue-button", "n_clicks"),
        Input("clear-queue-button", "n_clicks"),
        Input({"type": "remove-build-from-queue", "index": ALL}, "n_clicks"),
    ],
    [
        State("ablation-primary-store", "data"),
        State("full-sweep-build-queue", "data"),
        State("primary-build-source", "value"),
        State("custom-primary-weapon-dropdown", "value"),
        State("custom-primary-abilities-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def manage_build_queue(
    add_clicks,
    clear_clicks,
    remove_clicks,
    primary_data,
    current_queue,
    build_source,
    custom_weapon_id,
    custom_abilities,
):
    """Manage the build queue for full sweep analysis."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Initialize queue if None
    if current_queue is None:
        current_queue = []

    # Handle clear queue
    if triggered_id == "clear-queue-button":
        return []

    # Handle add primary build
    if triggered_id == "add-primary-to-queue-button":
        inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
        inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
        weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)

        # Determine build source
        if build_source == "custom":
            if not custom_weapon_id:
                return current_queue

            weapon_id = custom_weapon_id
            ability_tokens = custom_abilities if custom_abilities else []
        else:
            if not primary_data:
                return current_queue

            weapon_id = primary_data.get("weapon_id_token")
            ability_token_ids = primary_data.get("ability_input_tokens", [])

            if weapon_id is None:
                return current_queue

            # Convert token IDs to names
            ability_tokens = [
                str(inv_vocab.get(token_id, f"Unknown_{token_id}"))
                for token_id in ability_token_ids
            ]

        # Get weapon name
        weapon_raw = inv_weapon_vocab.get(weapon_id, f"Unknown_{weapon_id}")
        weapon_name = weapon_name_mapping.get(weapon_id, weapon_raw)

        # Create description
        if ability_tokens:
            abilities_str = ", ".join(ability_tokens[:3])
            if len(ability_tokens) > 3:
                abilities_str += f"... (+{len(ability_tokens) - 3})"
        else:
            abilities_str = "(empty)"
        description = f"{abilities_str} + {weapon_name}"

        # Add to queue (as dict for JSON serialization)
        new_build = {
            "weapon_id": weapon_id,
            "weapon_name": weapon_name,
            "ability_tokens": ability_tokens,
            "description": description,
        }
        current_queue.append(new_build)
        return current_queue

    # Handle remove build (pattern matching callback)
    if "{" in triggered_id:
        try:
            trigger_dict = json.loads(triggered_id)
            if trigger_dict.get("type") == "remove-build-from-queue":
                # Only process if the button was actually clicked (n_clicks is not None)
                # New buttons appearing trigger this callback with n_clicks=None
                trigger_value = ctx.triggered[0].get("value")
                if trigger_value is None:
                    return current_queue

                index = trigger_dict.get("index")
                if index is not None and 0 <= index < len(current_queue):
                    current_queue.pop(index)
                    return current_queue
        except json.JSONDecodeError:
            pass

    return current_queue


@dash.callback(
    Output("full-sweep-queue-display", "children"),
    Input("full-sweep-build-queue", "data"),
)
def display_build_queue(queue_data):
    """Display the current build queue."""
    if not queue_data:
        return html.P(
            "No builds in queue. Add builds from Primary Build or Top Examples.",
            className="text-muted mb-0 small",
        )

    items = []
    for i, build in enumerate(queue_data):
        items.append(
            dbc.Row(
                [
                    dbc.Col(
                        html.Span(
                            f"{i + 1}. {build.get('description', 'Unknown build')}",
                            className="small",
                        ),
                        width=10,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "×",
                            id={"type": "remove-build-from-queue", "index": i},
                            color="link",
                            size="sm",
                            className="text-danger p-0",
                        ),
                        width=2,
                        className="text-end",
                    ),
                ],
                className="mb-1",
            )
        )

    return html.Div(items)


@dash.callback(
    [
        Output("build-token-statistics-display", "children"),
        Output("build-token-statistics-collapse", "is_open"),
    ],
    Input("full-sweep-build-queue", "data"),
)
def compute_token_statistics(queue_data):
    """Compute and display token statistics across all builds in the queue."""
    if not queue_data or len(queue_data) == 0:
        return "", False

    # Parse all tokens from all builds
    # family_data[family_name] = {
    #     "builds": set of build indices with this family,
    #     "ap_counts": {ap_value: count}
    # }
    family_data: dict[str, dict] = {}
    total_builds = len(queue_data)

    for build_idx, build in enumerate(queue_data):
        ability_tokens = build.get("ability_tokens", [])
        seen_families_in_build = set()

        for token in ability_tokens:
            if not token or token.startswith("<"):
                continue

            family, ap_value = parse_token(token)

            if family not in family_data:
                family_data[family] = {"builds": set(), "ap_counts": {}}

            family_data[family]["builds"].add(build_idx)
            seen_families_in_build.add(family)

            if ap_value is not None:
                if ap_value not in family_data[family]["ap_counts"]:
                    family_data[family]["ap_counts"][ap_value] = 0
                family_data[family]["ap_counts"][ap_value] += 1

    if not family_data:
        return (
            html.P("No ability tokens in builds.", className="text-muted"),
            True,
        )

    # Sort families by number of builds (descending)
    sorted_families = sorted(
        family_data.items(),
        key=lambda x: len(x[1]["builds"]),
        reverse=True,
    )

    # Build the display
    stats_items = []
    for family, data in sorted_families:
        build_count = len(data["builds"])
        ap_counts = data["ap_counts"]

        # Format family name nicely
        family_display = family.replace("_", " ").title()

        # Build AP breakdown string
        if ap_counts:
            ap_parts = []
            for ap_val in sorted(ap_counts.keys()):
                ap_parts.append(f"_{ap_val}: {ap_counts[ap_val]}")
            ap_str = ", ".join(ap_parts)
        else:
            ap_str = "(main-only)"

        stats_items.append(
            html.Div(
                [
                    html.Strong(f"{family_display}: "),
                    html.Span(
                        f"{build_count}/{total_builds} builds",
                        className="text-primary me-2",
                    ),
                    html.Span(f"[{ap_str}]", className="text-muted"),
                ],
                className="mb-1",
            )
        )

    return html.Div(stats_items), True


def run_full_sweep_for_build(
    build: dict,
    feature_id: int,
    all_tokens: list[str],
    all_weapon_ids: list[int],
    weapon_properties_df,
    inv_weapon_vocab: dict,
    weapon_name_mapping: dict,
    build_idx: int,
) -> FullSweepResults | None:
    """Run full sweep for a single build."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    base_tokens = build.get("ability_tokens", [])
    weapon_id = build.get("weapon_id")
    weapon_raw = inv_weapon_vocab.get(weapon_id, f"Unknown_{weapon_id}")

    # If base has no tokens, use <NULL>
    if not base_tokens:
        base_tokens = ["<NULL>"]

    # Get feature name
    feature_name = f"Feature {feature_id}"
    if (
        hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        feature_name = (
            DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
                feature_id
            )
        )

    # Compute base activation
    base_activation = compute_feature_activation(
        base_tokens, weapon_raw, feature_id
    )
    if base_activation is None:
        return None

    logger.info(
        f"Build {build_idx + 1}: Base activation = {base_activation:.4f}"
    )

    # --- Token Sweep ---
    # For each token, we strip tokens of the same family from the base before testing.
    # This gives the isolated effect of adding that family.
    # IMPORTANT: Delta is computed relative to the STRIPPED base, not the original.
    family_deltas: dict[str, dict[int, float]] = {}  # family -> {ap: delta}
    main_only_results: list[MainOnlyResult] = []

    # Cache stripped base activations per family (for efficiency)
    stripped_base_cache: dict[str, tuple[list[str], float]] = {}

    for i, token in enumerate(all_tokens):
        if (i + 1) % 20 == 0:
            logger.info(f"  Testing token {i + 1}/{len(all_tokens)}: {token}")

        # Parse the token to get its family
        family, ap_value = parse_token(token)

        # Determine cache key (family for standard abilities, token for main-only)
        cache_key = family if ap_value is not None else token

        # Get or compute stripped base and its activation
        if cache_key in stripped_base_cache:
            stripped_base, stripped_base_activation = stripped_base_cache[
                cache_key
            ]
        else:
            # Strip tokens of this family from base (for isolated testing)
            if ap_value is not None:
                # Standard ability - remove all tokens of same family from base
                stripped_base = [
                    t for t in base_tokens if parse_token(t)[0] != family
                ]
            else:
                # Main-only ability - remove exact match only
                stripped_base = [t for t in base_tokens if t != token]

            # If no tokens remain after stripping, use <NULL>
            if not stripped_base:
                stripped_base = ["<NULL>"]

            # Compute activation for the stripped base
            stripped_base_activation = compute_feature_activation(
                stripped_base, weapon_raw, feature_id
            )
            if stripped_base_activation is None:
                stripped_base_activation = 0.0

            stripped_base_cache[cache_key] = (
                stripped_base,
                stripped_base_activation,
            )

        # Compute activation with stripped base + test token
        test_tokens = stripped_base + [token]
        test_activation = compute_feature_activation(
            test_tokens, weapon_raw, feature_id
        )

        if test_activation is None:
            continue

        # Delta is relative to STRIPPED base (not original base)
        delta = test_activation - stripped_base_activation

        # family and ap_value already parsed above
        if ap_value is not None:
            # Standard ability with AP value
            if family not in family_deltas:
                family_deltas[family] = {}
            family_deltas[family][ap_value] = delta
        else:
            # Main-only ability
            main_only_results.append(
                MainOnlyResult(ability_name=token, delta=delta)
            )

    # Convert family_deltas to AbilityFamilyResult list
    ability_families = []
    for family, ap_dict in sorted(family_deltas.items()):
        sorted_aps = sorted(ap_dict.keys())
        ability_families.append(
            AbilityFamilyResult(
                family_name=family,
                ap_values=sorted_aps,
                deltas=[ap_dict[ap] for ap in sorted_aps],
            )
        )

    # Sort main-only by absolute delta
    main_only_results.sort(key=lambda x: abs(x.delta), reverse=True)

    # --- Weapon Sweep ---
    logger.info(f"  Testing {len(all_weapon_ids)} weapons...")

    weapon_deltas: dict[int, tuple[str, float]] = (
        {}
    )  # weapon_id -> (name, delta)

    for i, test_weapon_id in enumerate(all_weapon_ids):
        if (i + 1) % 30 == 0:
            logger.info(f"  Testing weapon {i + 1}/{len(all_weapon_ids)}")

        test_weapon_raw = inv_weapon_vocab.get(
            test_weapon_id, f"Unknown_{test_weapon_id}"
        )
        test_weapon_name = weapon_name_mapping.get(
            test_weapon_id, test_weapon_raw
        )

        if test_weapon_id == weapon_id:
            # Current weapon, delta is 0
            weapon_deltas[test_weapon_id] = (test_weapon_name, 0.0)
        else:
            test_activation = compute_feature_activation(
                base_tokens, test_weapon_raw, feature_id
            )
            if test_activation is not None:
                delta = test_activation - base_activation
                weapon_deltas[test_weapon_id] = (test_weapon_name, delta)

    # Group weapons by special, sub, class
    by_special: dict[str, list[tuple[str, float]]] = {}
    by_sub: dict[str, list[tuple[str, float]]] = {}
    by_class: dict[str, list[tuple[str, float]]] = {}

    for wid, (wname, delta) in weapon_deltas.items():
        # Look up weapon properties using the raw weapon name from inv_weapon_vocab
        # The raw name is already in format "weapon_id_1234"
        weapon_id_str = inv_weapon_vocab.get(wid, f"weapon_id_{wid}")
        props_row = weapon_properties_df.filter(
            weapon_properties_df["weapon_id"] == weapon_id_str
        )

        if len(props_row) > 0:
            special = props_row["special"][0]
            sub = props_row["sub"][0]
            wclass = props_row["class"][0]

            if special not in by_special:
                by_special[special] = []
            by_special[special].append((wname, delta))

            if sub not in by_sub:
                by_sub[sub] = []
            by_sub[sub].append((wname, delta))

            if wclass not in by_class:
                by_class[wclass] = []
            by_class[wclass].append((wname, delta))

    # Convert to WeaponGroupResult
    def to_group_results(group_dict: dict) -> list[WeaponGroupResult]:
        results = []
        for group_name, weapons in group_dict.items():
            names = [w[0] for w in weapons]
            deltas = [w[1] for w in weapons]
            avg = sum(deltas) / len(deltas) if deltas else 0.0
            results.append(
                WeaponGroupResult(
                    group_name=group_name,
                    weapon_names=names,
                    deltas=deltas,
                    avg_delta=avg,
                )
            )
        # Sort by absolute average delta
        results.sort(key=lambda x: abs(x.avg_delta), reverse=True)
        return results

    # NULL base is computed once separately (not per-build)
    # null_base_* fields will use defaults from dataclass
    return FullSweepResults(
        build_id=build_idx,
        build_description=build.get("description", ""),
        feature_id=feature_id,
        feature_name=feature_name,
        base_activation=base_activation,
        ability_families=ability_families,
        main_only_abilities=main_only_results,
        by_special=to_group_results(by_special),
        by_sub=to_group_results(by_sub),
        by_class=to_group_results(by_class),
    )


def run_null_base_sweep(
    feature_id: int,
    all_weapon_ids: list[int],
    weapon_properties_df,
    inv_weapon_vocab: dict,
    weapon_name_mapping: dict,
    activation_threshold: float = 0.0,
) -> tuple[
    float,
    list[WeaponGroupResult],
    list[WeaponGroupResult],
    list[WeaponGroupResult],
    list[tuple[str, float, str, str, str]],
]:
    """
    Run NULL base weapon sweep once (independent of build abilities).

    Args:
        activation_threshold: Filter out weapons where |activation| < threshold.
                              Set to 0.01 * mean_base_activation to keep only
                              weapons with >= 1% of the base activation.

    Returns:
        (null_base_activation, by_special, by_sub, by_class, individual_weapons)
        where individual_weapons is list of (weapon_name, activation, special, sub, class) tuples
    """
    logger.info(
        f"Running NULL base weapon sweep for {len(all_weapon_ids)} weapons..."
    )
    logger.info(f"  Activation threshold: {activation_threshold:.4f}")

    # Compute raw activation for each weapon with NULL base
    # Store raw activations (not deltas) for filtering
    null_weapon_activations: dict[int, tuple[str, float]] = {}

    for test_weapon_id in all_weapon_ids:
        test_weapon_raw = inv_weapon_vocab.get(
            test_weapon_id, f"Unknown_{test_weapon_id}"
        )
        test_weapon_name = weapon_name_mapping.get(
            test_weapon_id, test_weapon_raw
        )

        test_activation = compute_feature_activation(
            ["<NULL>"], test_weapon_raw, feature_id
        )
        if test_activation is not None:
            # Filter: only keep weapons where |activation| >= threshold
            if abs(test_activation) >= activation_threshold:
                null_weapon_activations[test_weapon_id] = (
                    test_weapon_name,
                    test_activation,
                )

    # Use mean activation as the "null base activation" for display
    if null_weapon_activations:
        all_activations = [a for _, a in null_weapon_activations.values()]
        null_base_activation = sum(all_activations) / len(all_activations)
    else:
        null_base_activation = 0.0

    logger.info(
        f"  Kept {len(null_weapon_activations)}/{len(all_weapon_ids)} weapons "
        f"above threshold"
    )

    # Group NULL base weapons by special, sub, class
    # Also build individual weapons list with properties
    null_by_special: dict[str, list[tuple[str, float]]] = {}
    null_by_sub: dict[str, list[tuple[str, float]]] = {}
    null_by_class: dict[str, list[tuple[str, float]]] = {}
    # Store (name, activation, special, sub, class) for individual weapons
    individual_weapons_with_props: list[tuple[str, float, str, str, str]] = []

    for wid, (wname, activation) in null_weapon_activations.items():
        weapon_id_str = inv_weapon_vocab.get(wid, f"weapon_id_{wid}")
        props_row = weapon_properties_df.filter(
            weapon_properties_df["weapon_id"] == weapon_id_str
        )

        if len(props_row) > 0:
            special = props_row["special"][0]
            sub = props_row["sub"][0]
            wclass = props_row["class"][0]

            if special not in null_by_special:
                null_by_special[special] = []
            null_by_special[special].append((wname, activation))

            if sub not in null_by_sub:
                null_by_sub[sub] = []
            null_by_sub[sub].append((wname, activation))

            if wclass not in null_by_class:
                null_by_class[wclass] = []
            null_by_class[wclass].append((wname, activation))

            # Add to individual weapons list with properties
            individual_weapons_with_props.append(
                (wname, activation, special, sub, wclass)
            )

    # Convert to WeaponGroupResult
    def to_group_results(group_dict: dict) -> list[WeaponGroupResult]:
        results = []
        for group_name, weapons in group_dict.items():
            names = [w[0] for w in weapons]
            deltas = [w[1] for w in weapons]
            avg = sum(deltas) / len(deltas) if deltas else 0.0
            results.append(
                WeaponGroupResult(
                    group_name=group_name,
                    weapon_names=names,
                    deltas=deltas,
                    avg_delta=avg,
                )
            )
        results.sort(key=lambda x: abs(x.avg_delta), reverse=True)
        return results

    # Sort individual weapons by activation (descending)
    individual_weapons_with_props.sort(key=lambda x: x[1], reverse=True)

    logger.info("NULL base weapon sweep complete.")
    return (
        null_base_activation,
        to_group_results(null_by_special),
        to_group_results(null_by_sub),
        to_group_results(null_by_class),
        individual_weapons_with_props,
    )


def aggregate_sweep_results(
    all_results: list[FullSweepResults],
    method: Literal["mean", "median"] = "mean",
) -> FullSweepResults:
    """Aggregate results from multiple builds using mean or median."""
    if len(all_results) == 1:
        return all_results[0]

    agg_fn = statistics.mean if method == "mean" else statistics.median

    # Aggregate ability families
    family_data: dict[str, dict[int, list[float]]] = {}
    for result in all_results:
        for family in result.ability_families:
            if family.family_name not in family_data:
                family_data[family.family_name] = {}
            for ap, delta in zip(family.ap_values, family.deltas):
                if ap not in family_data[family.family_name]:
                    family_data[family.family_name][ap] = []
                family_data[family.family_name][ap].append(delta)

    agg_families = []
    for family_name, ap_dict in sorted(family_data.items()):
        sorted_aps = sorted(ap_dict.keys())
        agg_deltas = [agg_fn(ap_dict[ap]) for ap in sorted_aps]
        agg_families.append(
            AbilityFamilyResult(
                family_name=family_name,
                ap_values=sorted_aps,
                deltas=agg_deltas,
            )
        )

    # Aggregate main-only abilities
    main_only_data: dict[str, list[float]] = {}
    for result in all_results:
        for mo in result.main_only_abilities:
            if mo.ability_name not in main_only_data:
                main_only_data[mo.ability_name] = []
            main_only_data[mo.ability_name].append(mo.delta)

    agg_main_only = [
        MainOnlyResult(ability_name=name, delta=agg_fn(deltas))
        for name, deltas in main_only_data.items()
    ]
    agg_main_only.sort(key=lambda x: abs(x.delta), reverse=True)

    # Aggregate weapon groups
    def agg_weapon_groups(
        all_groups: list[list[WeaponGroupResult]],
    ) -> list[WeaponGroupResult]:
        group_data: dict[str, list[float]] = {}
        for groups in all_groups:
            for g in groups:
                if g.group_name not in group_data:
                    group_data[g.group_name] = []
                group_data[g.group_name].append(g.avg_delta)

        results = [
            WeaponGroupResult(
                group_name=name,
                weapon_names=[],  # Not tracking individual weapons in aggregation
                deltas=[],
                avg_delta=agg_fn(deltas),
            )
            for name, deltas in group_data.items()
        ]
        results.sort(key=lambda x: abs(x.avg_delta), reverse=True)
        return results

    agg_by_special = agg_weapon_groups([r.by_special for r in all_results])
    agg_by_sub = agg_weapon_groups([r.by_sub for r in all_results])
    agg_by_class = agg_weapon_groups([r.by_class for r in all_results])

    # Average base activation
    avg_base = agg_fn([r.base_activation for r in all_results])

    # NULL base fields are computed once separately and set by caller
    return FullSweepResults(
        build_id=-1,
        build_description=f"Aggregated ({len(all_results)} builds, {method})",
        feature_id=all_results[0].feature_id,
        feature_name=all_results[0].feature_name,
        base_activation=avg_base,
        ability_families=agg_families,
        main_only_abilities=agg_main_only,
        by_special=agg_by_special,
        by_sub=agg_by_sub,
        by_class=agg_by_class,
    )


def build_ability_family_graph(results: list[AbilityFamilyResult]) -> go.Figure:
    """Build line graph with one line per ability family."""
    fig = go.Figure()

    for family in results:
        fig.add_trace(
            go.Scatter(
                x=family.ap_values,
                y=family.deltas,
                mode="lines+markers",
                name=family.family_name,
                hovertemplate=f"{family.family_name}<br>AP: %{{x}}<br>Delta: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Ability Family Effects by AP Value",
        xaxis_title="AP Value",
        yaxis_title="Activation Delta",
        hovermode="closest",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=60, b=120),
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def build_main_only_bar_chart(results: list[MainOnlyResult]) -> go.Figure:
    """Build horizontal bar chart for main-only abilities."""
    if not results:
        return go.Figure()

    # Sort by delta value
    sorted_results = sorted(results, key=lambda x: x.delta)

    names = [r.ability_name for r in sorted_results]
    deltas = [r.delta for r in sorted_results]
    colors = ["green" if d > 0 else "red" for d in deltas]

    fig = go.Figure(
        go.Bar(
            x=deltas,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Delta: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Main-Only Ability Effects",
        xaxis_title="Activation Delta",
        yaxis_title="",
        height=max(200, len(results) * 25 + 100),
        margin=dict(l=150, r=40, t=60, b=40),
    )

    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def build_weapon_group_bar_chart(
    results: list[WeaponGroupResult],
    title: str,
    y_axis_label: str = "Avg Activation Delta",
    show_zero_line: bool = True,
) -> go.Figure:
    """Build bar chart for weapon grouping."""
    if not results:
        return go.Figure()

    # Sort by avg_delta (or avg activation for NULL base)
    sorted_results = sorted(results, key=lambda x: x.avg_delta, reverse=True)

    names = [r.group_name for r in sorted_results]
    values = [r.avg_delta for r in sorted_results]
    colors = ["green" if v > 0 else "red" for v in values]

    # Use appropriate hover label
    hover_label = (
        "Avg Activation" if "Activation" in y_axis_label else "Avg Delta"
    )

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            hovertemplate=f"%{{x}}<br>{hover_label}: %{{y:.4f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_axis_label,
        height=300,
        margin=dict(l=40, r=40, t=60, b=80),
        xaxis_tickangle=-45,
    )

    # Add zero line (optional - useful for deltas, less so for raw activations)
    if show_zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def format_full_sweep_markdown(results: FullSweepResults) -> str:
    """Format full sweep results as Markdown for clipboard export."""
    lines = [
        f"# Full Sweep Analysis: {results.feature_name}",
        "",
        f"**Build:** {results.build_description}",
        f"**Base Activation:** {results.base_activation:.4f}",
        "",
    ]

    # Ability families
    if results.ability_families:
        lines.append("## Ability Families by AP")
        lines.append("")
        lines.append(
            "| Ability | "
            + " | ".join(str(ap) for ap in [3, 6, 12, 15, 21, 29, 38, 51, 57])
            + " |"
        )
        lines.append("|" + "----|" * 10)

        for family in results.ability_families:
            ap_map = dict(zip(family.ap_values, family.deltas))
            row = [family.family_name]
            for ap in [3, 6, 12, 15, 21, 29, 38, 51, 57]:
                if ap in ap_map:
                    row.append(f"{ap_map[ap]:+.3f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Main-only abilities
    if results.main_only_abilities:
        lines.append("## Main-Only Abilities")
        lines.append("")
        lines.append("| Ability | Delta |")
        lines.append("|---------|-------|")
        for mo in results.main_only_abilities:
            lines.append(f"| {mo.ability_name} | {mo.delta:+.4f} |")
        lines.append("")

    # Weapon groups
    for group_name, groups in [
        ("By Special", results.by_special),
        ("By Sub Weapon", results.by_sub),
        ("By Weapon Class", results.by_class),
    ]:
        if groups:
            lines.append(f"## Weapons {group_name}")
            lines.append("")
            lines.append("| Group | Avg Delta |")
            lines.append("|-------|-----------|")
            for g in groups:
                lines.append(f"| {g.group_name} | {g.avg_delta:+.4f} |")
            lines.append("")

    # NULL base weapon groups (weapon effects with no abilities)
    # These show raw activation values, not deltas
    null_base_groups = [
        ("By Special (NULL Base)", results.null_base_by_special),
        ("By Sub Weapon (NULL Base)", results.null_base_by_sub),
        ("By Weapon Class (NULL Base)", results.null_base_by_class),
    ]
    if any(groups for _, groups in null_base_groups):
        lines.append("---")
        lines.append("")
        lines.append("## Weapon Effects (NULL Base - No Abilities)")
        lines.append("")
        lines.append(
            f"**{len(results.null_base_weapons)} weapons** with activation >= 1% of base. "
            f"Mean: {results.null_base_activation:.4f}"
        )
        lines.append("")

        # Individual weapons table
        if results.null_base_weapons:
            lines.append("### Individual Weapon Kits")
            lines.append("")
            lines.append("| Weapon | Special | Sub | Class | Activation |")
            lines.append("|--------|---------|-----|-------|------------|")
            for (
                name,
                activation,
                special,
                sub,
                wclass,
            ) in results.null_base_weapons:
                lines.append(
                    f"| {name} | {special} | {sub} | {wclass} | {activation:.4f} |"
                )
            lines.append("")

        # Grouped summaries
        for group_name, groups in null_base_groups:
            if groups:
                lines.append(f"### {group_name}")
                lines.append("")
                lines.append("| Group | Avg Activation |")
                lines.append("|-------|----------------|")
                for g in groups:
                    lines.append(f"| {g.group_name} | {g.avg_delta:.4f} |")
                lines.append("")

    return "\n".join(lines)


def build_full_sweep_display(results: FullSweepResults) -> html.Div:
    """Build the complete display for full sweep results."""
    markdown_text = format_full_sweep_markdown(results)

    sections = [
        # Header with copy button
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5(results.build_description, className="mb-1"),
                        html.P(
                            f"Base Activation: {results.base_activation:.4f}",
                            className="text-muted mb-0",
                        ),
                    ],
                    width=10,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Clipboard(
                                    target_id="full-sweep-markdown-content",
                                    title="Copy to clipboard (Markdown)",
                                    className="btn btn-outline-secondary btn-sm",
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
                    width=2,
                    className="d-flex align-items-center justify-content-center",
                ),
            ],
            className="mb-3",
        ),
        # Hidden div with markdown content
        html.Div(
            markdown_text,
            id="full-sweep-markdown-content",
            style={"display": "none"},
        ),
    ]

    # Ability family graph
    if results.ability_families:
        sections.append(
            dcc.Graph(
                figure=build_ability_family_graph(results.ability_families),
                config={"displayModeBar": False},
            )
        )

    # Main-only bar chart
    if results.main_only_abilities:
        sections.append(
            dcc.Graph(
                figure=build_main_only_bar_chart(results.main_only_abilities),
                config={"displayModeBar": False},
            )
        )

    # Weapon group charts in a row
    weapon_charts = []
    if results.by_special:
        weapon_charts.append(
            dbc.Col(
                dcc.Graph(
                    figure=build_weapon_group_bar_chart(
                        results.by_special, "By Special"
                    ),
                    config={"displayModeBar": False},
                ),
                md=4,
            )
        )
    if results.by_sub:
        weapon_charts.append(
            dbc.Col(
                dcc.Graph(
                    figure=build_weapon_group_bar_chart(
                        results.by_sub, "By Sub Weapon"
                    ),
                    config={"displayModeBar": False},
                ),
                md=4,
            )
        )
    if results.by_class:
        weapon_charts.append(
            dbc.Col(
                dcc.Graph(
                    figure=build_weapon_group_bar_chart(
                        results.by_class, "By Weapon Class"
                    ),
                    config={"displayModeBar": False},
                ),
                md=4,
            )
        )

    if weapon_charts:
        sections.append(dbc.Row(weapon_charts, className="mt-3"))

    # NULL base weapon charts (weapon effects with no abilities)
    # These show raw activation values (filtered to >= 1% of base activation)
    if (
        results.null_base_by_special
        or results.null_base_by_sub
        or results.null_base_by_class
    ):
        sections.append(
            html.Hr(className="my-4"),
        )
        sections.append(
            html.H5(
                "Weapon Effects (NULL Base - No Abilities)",
                className="mt-3 mb-2",
            ),
        )
        sections.append(
            html.P(
                f"Showing {len(results.null_base_weapons)} weapons with activation >= 1% of base. "
                f"Mean: {results.null_base_activation:.4f}",
                className="text-muted mb-3",
            ),
        )

        # Individual weapons table (collapsible)
        if results.null_base_weapons:
            weapon_rows = [
                html.Tr(
                    [
                        html.Td(name, style={"fontSize": "0.85rem"}),
                        html.Td(special, style={"fontSize": "0.85rem"}),
                        html.Td(sub, style={"fontSize": "0.85rem"}),
                        html.Td(wclass, style={"fontSize": "0.85rem"}),
                        html.Td(
                            f"{activation:.4f}",
                            style={"fontSize": "0.85rem", "textAlign": "right"},
                        ),
                    ]
                )
                for name, activation, special, sub, wclass in results.null_base_weapons
            ]

            sections.append(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            dbc.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Weapon Kit"),
                                                html.Th("Special"),
                                                html.Th("Sub"),
                                                html.Th("Class"),
                                                html.Th(
                                                    "Activation",
                                                    style={
                                                        "textAlign": "right"
                                                    },
                                                ),
                                            ]
                                        )
                                    ),
                                    html.Tbody(weapon_rows),
                                ],
                                striped=True,
                                bordered=True,
                                hover=True,
                                size="sm",
                                style={
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                },
                            ),
                            title=f"Individual Weapon Kits ({len(results.null_base_weapons)} weapons)",
                        ),
                    ],
                    start_collapsed=True,
                    className="mb-3",
                )
            )

        null_weapon_charts = []
        if results.null_base_by_special:
            null_weapon_charts.append(
                dbc.Col(
                    dcc.Graph(
                        figure=build_weapon_group_bar_chart(
                            results.null_base_by_special,
                            "By Special (NULL Base)",
                            y_axis_label="Avg Activation",
                            show_zero_line=False,
                        ),
                        config={"displayModeBar": False},
                    ),
                    md=4,
                )
            )
        if results.null_base_by_sub:
            null_weapon_charts.append(
                dbc.Col(
                    dcc.Graph(
                        figure=build_weapon_group_bar_chart(
                            results.null_base_by_sub,
                            "By Sub Weapon (NULL Base)",
                            y_axis_label="Avg Activation",
                            show_zero_line=False,
                        ),
                        config={"displayModeBar": False},
                    ),
                    md=4,
                )
            )
        if results.null_base_by_class:
            null_weapon_charts.append(
                dbc.Col(
                    dcc.Graph(
                        figure=build_weapon_group_bar_chart(
                            results.null_base_by_class,
                            "By Weapon Class (NULL Base)",
                            y_axis_label="Avg Activation",
                            show_zero_line=False,
                        ),
                        config={"displayModeBar": False},
                    ),
                    md=4,
                )
            )

        if null_weapon_charts:
            sections.append(dbc.Row(null_weapon_charts, className="mt-3"))

    return html.Div(sections)


@dash.callback(
    Output("full-sweep-results-display", "children"),
    Input("run-full-sweep-button", "n_clicks"),
    State("full-sweep-build-queue", "data"),
    State("feature-dropdown", "value"),
    State("full-sweep-aggregation-method", "value"),
    prevent_initial_call=True,
)
def run_full_sweep_callback(
    n_clicks, queue_data, feature_id, aggregation_method
):
    """Run the full sweep analysis."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return html.P(
            "Add builds to the queue and click 'Run Full Sweep'.",
            className="text-muted",
        )

    if not queue_data or len(queue_data) == 0:
        return dbc.Alert(
            "Please add at least one build to the queue.",
            color="warning",
        )

    if feature_id is None:
        return dbc.Alert(
            "Please select a feature from the dropdown.",
            color="warning",
        )

    # Get vocabs and weapon properties
    vocab = getattr(DASHBOARD_CONTEXT, "vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)

    # Get all tokens (excluding special tokens)
    all_tokens = [tok for tok in vocab.keys() if not tok.startswith("<")]
    all_weapon_ids = list(inv_weapon_vocab.keys())

    # Get weapon properties
    try:
        weapon_properties_df = get_weapon_properties_df()
    except Exception as e:
        logger.error(f"Failed to get weapon properties: {e}")
        return dbc.Alert(
            f"Failed to get weapon properties: {e}", color="danger"
        )

    logger.info(
        f"Starting full sweep: {len(queue_data)} builds, "
        f"{len(all_tokens)} tokens, {len(all_weapon_ids)} weapons"
    )

    # Run sweep for each build
    all_results = []
    for i, build in enumerate(queue_data):
        logger.info(
            f"Processing build {i + 1}/{len(queue_data)}: {build.get('description', '')}"
        )
        result = run_full_sweep_for_build(
            build=build,
            feature_id=feature_id,
            all_tokens=all_tokens,
            all_weapon_ids=all_weapon_ids,
            weapon_properties_df=weapon_properties_df,
            inv_weapon_vocab=inv_weapon_vocab,
            weapon_name_mapping=weapon_name_mapping,
            build_idx=i,
        )
        if result:
            all_results.append(result)

    if not all_results:
        return dbc.Alert(
            "Failed to compute sweep for any builds. Check logs for details.",
            color="danger",
        )

    logger.info(
        f"Full sweep complete. Aggregating {len(all_results)} results..."
    )

    # Aggregate results
    aggregated = aggregate_sweep_results(all_results, aggregation_method)

    # Compute activation threshold: 1% of mean base activation
    # Weapons with |activation| < threshold will be filtered out
    activation_threshold = 0.01 * abs(aggregated.base_activation)
    logger.info(
        f"NULL base threshold: 1% of {aggregated.base_activation:.4f} = "
        f"{activation_threshold:.4f}"
    )

    # Run NULL base weapon sweep once (independent of builds)
    (
        null_base_activation,
        null_base_by_special,
        null_base_by_sub,
        null_base_by_class,
        null_base_weapons,
    ) = run_null_base_sweep(
        feature_id=feature_id,
        all_weapon_ids=all_weapon_ids,
        weapon_properties_df=weapon_properties_df,
        inv_weapon_vocab=inv_weapon_vocab,
        weapon_name_mapping=weapon_name_mapping,
        activation_threshold=activation_threshold,
    )

    # Set NULL base fields on the aggregated result
    aggregated.null_base_activation = null_base_activation
    aggregated.null_base_by_special = null_base_by_special
    aggregated.null_base_by_sub = null_base_by_sub
    aggregated.null_base_by_class = null_base_by_class
    aggregated.null_base_weapons = null_base_weapons

    return build_full_sweep_display(aggregated)


# ---------------------------------------------------------------------------
# Higher-Order Interaction Analysis Functions
# ---------------------------------------------------------------------------


def get_family_name(token: str) -> str | None:
    """
    Extract family name from a token.

    Examples:
        "swim_speed_up_21" -> "swim_speed_up"
        "ninja_squid" -> "ninja_squid"
        "<PAD>" -> None
    """
    if token.startswith("<"):
        return None

    match = ABILITY_FAMILY_RE.match(token)
    if match:
        return match.group(1)

    # Main-only abilities don't have AP suffix
    from splatnlp.utils.constants import MAIN_ONLY_ABILITIES

    if token in MAIN_ONLY_ABILITIES:
        return token

    return token  # Return as-is if no pattern match


def compute_contrastive_itemsets(
    feature_id: int,
    high_pct: float = 5.0,
    family_mode: bool = False,
) -> ContrastiveItemsetResults | None:
    """
    Mine frequent itemsets in high-activation examples vs baseline.

    Args:
        feature_id: SAE feature to analyze
        high_pct: Top percentile for "high" activation (e.g., 5.0 = top 5%)
        family_mode: If True, collapse tokens to ability families

    Returns:
        ContrastiveItemsetResults with itemsets sorted by lift
    """
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    logger.info(
        f"[Itemset Mining] Starting for feature {feature_id}, "
        f"top {high_pct}%, family_mode={family_mode}"
    )

    inv_vocab = DASHBOARD_CONTEXT.inv_vocab

    # Get all activations for this feature (cached)
    logger.info("[Itemset Mining] Loading activation data...")
    all_data = get_cached_activations(feature_id)

    if all_data is None or len(all_data) == 0:
        logger.warning("[Itemset Mining] No activation data found")
        return None

    # Sort by activation and compute percentile thresholds
    activations = all_data["activation"].to_list()
    logger.info(f"[Itemset Mining] Loaded {len(activations)} examples")
    n_total = len(activations)

    high_threshold_idx = int(n_total * (1 - high_pct / 100))
    mid_low_idx = int(n_total * 0.25)
    mid_high_idx = int(n_total * 0.75)

    sorted_indices = sorted(range(n_total), key=lambda i: activations[i])
    high_indices = set(sorted_indices[high_threshold_idx:])
    mid_indices = set(sorted_indices[mid_low_idx:mid_high_idx])

    # Extract token sets
    ability_tokens_list = all_data["ability_input_tokens"].to_list()

    def get_token_set(tokens):
        if family_mode:
            families = set()
            for tid in tokens:
                name = inv_vocab.get(tid, "")
                family = get_family_name(name)
                if family:
                    families.add(family)
            return frozenset(families)
        else:
            return frozenset(
                inv_vocab.get(t, "") for t in tokens if t in inv_vocab
            )

    high_sets = []
    mid_sets = []

    for idx in range(n_total):
        token_set = get_token_set(ability_tokens_list[idx])
        if len(token_set) < 2:
            continue
        if idx in high_indices:
            high_sets.append(token_set)
        elif idx in mid_indices:
            mid_sets.append(token_set)

    if len(high_sets) == 0 or len(mid_sets) == 0:
        logger.warning(
            "[Itemset Mining] Insufficient examples in high/mid groups"
        )
        return None

    logger.info(
        f"[Itemset Mining] Split: {len(high_sets)} high, {len(mid_sets)} baseline"
    )
    logger.info("[Itemset Mining] Counting itemsets of size 2, 3, 4...")

    # Count itemsets of size 2, 3, 4
    def count_itemsets(sets_list, max_size=4):
        counts = {2: Counter(), 3: Counter(), 4: Counter()}
        for s in sets_list:
            s_list = list(s)
            for size in [2, 3, 4]:
                if len(s_list) >= size:
                    for combo in combinations(s_list, size):
                        counts[size][tuple(sorted(combo))] += 1
        return counts

    high_counts = count_itemsets(high_sets)
    mid_counts = count_itemsets(mid_sets)

    n_high = len(high_sets)
    n_mid = len(mid_sets)

    # Compute lift for each itemset
    itemsets_by_size = {}
    min_count_high = max(2, int(n_high * 0.01))  # At least 1% support in high

    for size in [2, 3, 4]:
        itemsets = []
        for tokens, count_high in high_counts[size].most_common(200):
            if count_high < min_count_high:
                continue
            count_mid = mid_counts[size].get(tokens, 0)
            support_high = count_high / n_high
            support_mid = (count_mid / n_mid) if n_mid > 0 else 0.001

            # Avoid division by zero
            if support_mid < 0.001:
                support_mid = 0.001

            lift = support_high / support_mid

            if lift > 1.5:  # Only keep itemsets with meaningful lift
                itemsets.append(
                    FrequentItemset(
                        tokens=tokens,
                        support_high=support_high,
                        support_baseline=support_mid,
                        lift=lift,
                        count_high=count_high,
                    )
                )

        # Sort by lift
        itemsets.sort(key=lambda x: x.lift, reverse=True)
        itemsets_by_size[size] = itemsets[:50]  # Top 50 per size
        logger.info(
            f"[Itemset Mining] Size-{size}: found {len(itemsets)} itemsets with lift > 1.5"
        )

    total_itemsets = sum(len(v) for v in itemsets_by_size.values())
    logger.info(f"[Itemset Mining] Complete. Total itemsets: {total_itemsets}")

    return ContrastiveItemsetResults(
        feature_id=feature_id,
        high_threshold_pct=high_pct,
        n_high_examples=n_high,
        n_baseline_examples=n_mid,
        itemsets_by_size=itemsets_by_size,
    )


def compute_interaction_sweep(
    feature_id: int,
    candidate_tokens: list[str],
    n_contexts: int = 50,
    family_mode: bool = False,
    include_three_way: bool = False,
) -> InteractionSweepResults | None:
    """
    Compute pairwise (and optional 3-way) interaction scores.

    For each pair (i,j), computes:
        I_ij(B) = f(B∪{i,j}) - f(B∪{i}) - f(B∪{j}) + f(B)

    I > 0 means synergy, I < 0 means redundancy.

    Args:
        feature_id: SAE feature to analyze
        candidate_tokens: List of token names to test (10-50 recommended)
        n_contexts: Number of base contexts to sample from mid-activation examples
        family_mode: If True, use family-representative tokens
        include_three_way: If True, also compute 3-way interactions for top pairs

    Returns:
        InteractionSweepResults with pairwise (and optional 3-way) scores
    """
    import numpy as np

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    n_pairs = len(candidate_tokens) * (len(candidate_tokens) - 1) // 2
    logger.info(
        f"[Interaction Sweep] Starting for feature {feature_id}: "
        f"{len(candidate_tokens)} tokens = {n_pairs} pairs, "
        f"{n_contexts} contexts, 3-way={include_three_way}"
    )

    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    vocab = DASHBOARD_CONTEXT.vocab

    if len(candidate_tokens) < 2:
        logger.warning("[Interaction Sweep] Need at least 2 tokens")
        return None

    # Get mid-activation examples for base contexts (cached)
    logger.info("[Interaction Sweep] Loading activation data...")
    all_data = get_cached_activations(feature_id)

    if all_data is None or len(all_data) == 0:
        logger.warning("[Interaction Sweep] No activation data found")
        return None

    # Sort and get mid-range examples (25-75 percentile)
    n_total = len(all_data)
    activations = all_data["activation"].to_list()
    sorted_indices = sorted(range(n_total), key=lambda i: activations[i])
    mid_low = int(n_total * 0.25)
    mid_high = int(n_total * 0.75)
    mid_indices = sorted_indices[mid_low:mid_high]

    # Sample base contexts
    import random

    sampled_indices = random.sample(
        mid_indices, min(n_contexts, len(mid_indices))
    )

    ability_tokens_list = all_data["ability_input_tokens"].to_list()
    weapon_ids = all_data["weapon_id"].to_list()
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab

    base_contexts = []
    for idx in sampled_indices:
        tokens = [inv_vocab.get(t, "") for t in ability_tokens_list[idx]]
        tokens = [t for t in tokens if t and not t.startswith("<")]
        weapon = inv_weapon_vocab.get(weapon_ids[idx], "")
        if weapon and tokens:
            base_contexts.append((tokens, weapon))

    if len(base_contexts) < 10:
        logger.warning("[Interaction Sweep] Insufficient base contexts (<10)")
        return None

    logger.info(
        f"[Interaction Sweep] Sampled {len(base_contexts)} base contexts"
    )

    # Compute pairwise interactions
    pairwise_scores = []
    total_pairs = len(candidate_tokens) * (len(candidate_tokens) - 1) // 2
    pairs_done = 0
    last_log_pct = 0

    logger.info(
        f"[Interaction Sweep] Computing {total_pairs} pairwise interactions..."
    )

    for i, tok_i in enumerate(candidate_tokens):
        for j, tok_j in enumerate(candidate_tokens):
            if j <= i:
                continue

            interactions = []
            for base_tokens, weapon in base_contexts:
                # Skip if tokens already in base
                if tok_i in base_tokens or tok_j in base_tokens:
                    continue

                # Compute f(B), f(B+i), f(B+j), f(B+i+j)
                f_B = compute_feature_activation(
                    base_tokens, weapon, feature_id
                )
                f_Bi = compute_feature_activation(
                    base_tokens + [tok_i], weapon, feature_id
                )
                f_Bj = compute_feature_activation(
                    base_tokens + [tok_j], weapon, feature_id
                )
                f_Bij = compute_feature_activation(
                    base_tokens + [tok_i, tok_j], weapon, feature_id
                )

                if all(x is not None for x in [f_B, f_Bi, f_Bj, f_Bij]):
                    I_ij = f_Bij - f_Bi - f_Bj + f_B
                    interactions.append(I_ij)

            if len(interactions) >= 5:
                mean_I = np.mean(interactions)
                std_I = np.std(interactions) / np.sqrt(len(interactions))
                pairwise_scores.append(
                    InteractionScore(
                        tokens=(tok_i, tok_j),
                        interaction_value=float(mean_I),
                        std_error=float(std_I),
                        n_contexts=len(interactions),
                    )
                )

            pairs_done += 1
            pct_done = int(100 * pairs_done / total_pairs)
            if pct_done >= last_log_pct + 10:
                logger.info(
                    f"[Interaction Sweep] Progress: {pairs_done}/{total_pairs} "
                    f"pairs ({pct_done}%)"
                )
                last_log_pct = pct_done

    logger.info(
        f"[Interaction Sweep] Pairwise complete: {len(pairwise_scores)} valid scores"
    )

    # Sort by absolute interaction value
    pairwise_scores.sort(key=lambda x: abs(x.interaction_value), reverse=True)

    # Optional: compute 3-way for top synergistic pairs
    three_way_scores = None
    if include_three_way and len(pairwise_scores) >= 3:
        logger.info("[Interaction Sweep] Computing 3-way interactions...")
        # Get top 10 synergistic pairs (I > 0)
        top_synergy = [s for s in pairwise_scores if s.interaction_value > 0][
            :10
        ]
        if len(top_synergy) >= 2:
            three_way_scores = []
            # Test triangles formed by top pairs
            tested_triads = set()
            for score1 in top_synergy[:5]:
                for score2 in top_synergy[:5]:
                    if score1 is score2:
                        continue
                    # Find common token or third token
                    tokens1 = set(score1.tokens)
                    tokens2 = set(score2.tokens)
                    all_tokens = tokens1 | tokens2
                    if len(all_tokens) == 3:
                        triad = tuple(sorted(all_tokens))
                        if triad in tested_triads:
                            continue
                        tested_triads.add(triad)

                        tok_i, tok_j, tok_k = triad
                        interactions = []

                        for base_tokens, weapon in base_contexts[:30]:
                            if any(
                                t in base_tokens for t in [tok_i, tok_j, tok_k]
                            ):
                                continue

                            f_B = compute_feature_activation(
                                base_tokens, weapon, feature_id
                            )
                            f_i = compute_feature_activation(
                                base_tokens + [tok_i], weapon, feature_id
                            )
                            f_j = compute_feature_activation(
                                base_tokens + [tok_j], weapon, feature_id
                            )
                            f_k = compute_feature_activation(
                                base_tokens + [tok_k], weapon, feature_id
                            )
                            f_ij = compute_feature_activation(
                                base_tokens + [tok_i, tok_j], weapon, feature_id
                            )
                            f_ik = compute_feature_activation(
                                base_tokens + [tok_i, tok_k], weapon, feature_id
                            )
                            f_jk = compute_feature_activation(
                                base_tokens + [tok_j, tok_k], weapon, feature_id
                            )
                            f_ijk = compute_feature_activation(
                                base_tokens + [tok_i, tok_j, tok_k],
                                weapon,
                                feature_id,
                            )

                            vals = [f_B, f_i, f_j, f_k, f_ij, f_ik, f_jk, f_ijk]
                            if all(v is not None for v in vals):
                                # 3-way interaction term
                                I_ijk = (
                                    f_ijk
                                    - f_ij
                                    - f_ik
                                    - f_jk
                                    + f_i
                                    + f_j
                                    + f_k
                                    - f_B
                                )
                                interactions.append(I_ijk)

                        if len(interactions) >= 3:
                            three_way_scores.append(
                                InteractionScore(
                                    tokens=triad,
                                    interaction_value=float(
                                        np.mean(interactions)
                                    ),
                                    std_error=float(
                                        np.std(interactions)
                                        / np.sqrt(len(interactions))
                                    ),
                                    n_contexts=len(interactions),
                                )
                            )

            three_way_scores.sort(
                key=lambda x: abs(x.interaction_value), reverse=True
            )
            logger.info(
                f"[Interaction Sweep] 3-way complete: {len(three_way_scores)} triads"
            )

    logger.info("[Interaction Sweep] Complete")

    return InteractionSweepResults(
        feature_id=feature_id,
        family_mode=family_mode,
        candidate_tokens=candidate_tokens,
        n_base_contexts=len(base_contexts),
        pairwise=pairwise_scores,
        three_way=three_way_scores,
    )


def compute_minimal_activating_sets(
    feature_id: int,
    n_examples: int = 50,
    threshold_ratio: float = 0.8,
) -> MinimalSetResults | None:
    """
    Find minimal token subsets that maintain feature activation.

    Uses greedy backward elimination: repeatedly remove the token
    whose removal hurts activation least, until any removal would
    drop below threshold_ratio * original_activation.

    Args:
        feature_id: SAE feature to analyze
        n_examples: Number of top examples to analyze
        threshold_ratio: Minimum activation retention (e.g., 0.8 = 80%)

    Returns:
        MinimalSetResults with minimal sets and common cores
    """
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    logger.info(
        f"[Minimal Sets] Starting for feature {feature_id}: "
        f"{n_examples} examples, {threshold_ratio:.0%} threshold"
    )

    db = DASHBOARD_CONTEXT.db
    inv_vocab = DASHBOARD_CONTEXT.inv_vocab
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab

    # Get top activating examples
    logger.info("[Minimal Sets] Loading top examples...")
    try:
        top_data = db.get_feature_activations(feature_id, limit=n_examples * 2)
    except Exception as e:
        logger.error(f"Failed to get activations: {e}")
        return None

    if top_data is None or len(top_data) == 0:
        logger.warning("[Minimal Sets] No examples found")
        return None

    logger.info(f"[Minimal Sets] Loaded {len(top_data)} candidate examples")

    ability_tokens_list = top_data["ability_input_tokens"].to_list()
    weapon_ids = top_data["weapon_id_token"].to_list()
    activations = top_data["activation"].to_list()

    minimal_sets = []
    processed = 0
    last_log_pct = 0

    for idx in range(len(top_data)):
        if processed >= n_examples:
            break

        tokens = [inv_vocab.get(t, "") for t in ability_tokens_list[idx]]
        tokens = [t for t in tokens if t and not t.startswith("<")]
        weapon = inv_weapon_vocab.get(weapon_ids[idx], "")

        if not weapon or len(tokens) < 2:
            continue

        original_activation = activations[idx]
        threshold = original_activation * threshold_ratio

        current_tokens = list(tokens)
        current_activation = original_activation

        # Greedy backward elimination
        while len(current_tokens) > 1:
            best_removal = None
            best_activation_after = -float("inf")

            for i, tok in enumerate(current_tokens):
                test_tokens = current_tokens[:i] + current_tokens[i + 1 :]
                test_act = compute_feature_activation(
                    test_tokens, weapon, feature_id
                )

                if test_act is not None and test_act > best_activation_after:
                    best_activation_after = test_act
                    best_removal = i

            if best_removal is None:
                break

            if best_activation_after >= threshold:
                # Safe to remove
                current_tokens.pop(best_removal)
                current_activation = best_activation_after
            else:
                # Can't remove any more without dropping below threshold
                break

        minimal_sets.append(
            MinimalActivatingSet(
                original_tokens=tokens,
                minimal_tokens=current_tokens,
                weapon_name=weapon,
                original_activation=original_activation,
                minimal_activation=current_activation,
                retention_ratio=(
                    current_activation / original_activation
                    if original_activation > 0
                    else 0
                ),
            )
        )
        processed += 1

        # Log progress every 10%
        pct_done = int(100 * processed / n_examples)
        if pct_done >= last_log_pct + 10:
            logger.info(
                f"[Minimal Sets] Progress: {processed}/{n_examples} "
                f"examples ({pct_done}%)"
            )
            last_log_pct = pct_done

    if not minimal_sets:
        logger.warning("[Minimal Sets] No valid minimal sets found")
        return None

    # Find common cores (2-4 token patterns)
    core_counter = Counter()
    for ms in minimal_sets:
        tokens = tuple(sorted(ms.minimal_tokens))
        if 2 <= len(tokens) <= 6:
            # Count the full minimal set
            core_counter[tokens] += 1
            # Also count subsets of size 2-3
            for size in [2, 3]:
                if len(tokens) >= size:
                    for combo in combinations(tokens, size):
                        core_counter[tuple(sorted(combo))] += 1

    # Get most common cores
    common_cores = [
        (core, count)
        for core, count in core_counter.most_common(20)
        if count >= 2
    ]

    logger.info(
        f"[Minimal Sets] Complete: {len(minimal_sets)} sets, "
        f"{len(common_cores)} common cores"
    )

    return MinimalSetResults(
        feature_id=feature_id,
        threshold_ratio=threshold_ratio,
        n_examples_analyzed=len(minimal_sets),
        minimal_sets=minimal_sets,
        common_cores=common_cores,
    )


# ---------------------------------------------------------------------------
# Visualization Functions for Higher-Order Analysis
# ---------------------------------------------------------------------------


def build_itemset_display(results: ContrastiveItemsetResults) -> html.Div:
    """Build display for contrastive itemset results."""
    if results is None:
        return html.Div("No results to display.", className="text-muted")

    sections = []

    # Summary
    sections.append(
        dbc.Alert(
            [
                html.Strong("Analysis Summary: "),
                f"Top {results.high_threshold_pct}% = {results.n_high_examples} examples, "
                f"Baseline (25-75%) = {results.n_baseline_examples} examples",
            ],
            color="info",
            className="mb-3",
        )
    )

    # Tables for each size
    for size in [2, 3, 4]:
        itemsets = results.itemsets_by_size.get(size, [])
        if not itemsets:
            continue

        rows = []
        for item in itemsets[:20]:
            tokens_str = ", ".join(item.tokens)
            rows.append(
                html.Tr(
                    [
                        html.Td(tokens_str),
                        html.Td(
                            f"{item.lift:.2f}", style={"textAlign": "right"}
                        ),
                        html.Td(
                            f"{item.support_high:.1%}",
                            style={"textAlign": "right"},
                        ),
                        html.Td(
                            f"{item.support_baseline:.1%}",
                            style={"textAlign": "right"},
                        ),
                        html.Td(
                            str(item.count_high), style={"textAlign": "right"}
                        ),
                    ]
                )
            )

        sections.append(
            html.Div(
                [
                    html.H6(f"Size-{size} Itemsets (Top {len(itemsets)})"),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Tokens"),
                                        html.Th(
                                            "Lift", style={"textAlign": "right"}
                                        ),
                                        html.Th(
                                            "Support (High)",
                                            style={"textAlign": "right"},
                                        ),
                                        html.Th(
                                            "Support (Base)",
                                            style={"textAlign": "right"},
                                        ),
                                        html.Th(
                                            "Count",
                                            style={"textAlign": "right"},
                                        ),
                                    ]
                                )
                            ),
                            html.Tbody(rows),
                        ],
                        striped=True,
                        bordered=True,
                        hover=True,
                        size="sm",
                    ),
                ],
                className="mb-4",
            )
        )

    if not sections:
        return html.Div(
            "No significant itemsets found. Try lowering the threshold.",
            className="text-muted",
        )

    return html.Div(sections)


def build_interaction_heatmap(results: InteractionSweepResults) -> html.Div:
    """Build clustered heatmap for interaction sweep results."""
    if results is None or len(results.pairwise) == 0:
        return html.Div(
            "No interaction results to display.", className="text-muted"
        )

    import numpy as np

    sections = []

    # Summary
    sections.append(
        dbc.Alert(
            [
                html.Strong("Interaction Sweep: "),
                f"{len(results.candidate_tokens)} tokens, "
                f"{results.n_base_contexts} base contexts, "
                f"{len(results.pairwise)} pairs computed",
            ],
            color="info",
            className="mb-3",
        )
    )

    # Top synergies and redundancies table
    synergies = [s for s in results.pairwise if s.interaction_value > 0][:10]
    redundancies = [s for s in results.pairwise if s.interaction_value < 0][:10]

    def make_interaction_table(scores, title, color):
        if not scores:
            return html.Div()
        rows = []
        for s in scores:
            tokens_str = " + ".join(s.tokens)
            rows.append(
                html.Tr(
                    [
                        html.Td(tokens_str),
                        html.Td(
                            f"{s.interaction_value:+.4f}",
                            style={
                                "textAlign": "right",
                                "color": (
                                    "green"
                                    if s.interaction_value > 0
                                    else "red"
                                ),
                            },
                        ),
                        html.Td(
                            f"±{s.std_error:.4f}", style={"textAlign": "right"}
                        ),
                        html.Td(
                            str(s.n_contexts), style={"textAlign": "right"}
                        ),
                    ]
                )
            )
        return html.Div(
            [
                html.H6(title, style={"color": color}),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Token Pair"),
                                    html.Th("I", style={"textAlign": "right"}),
                                    html.Th("SE", style={"textAlign": "right"}),
                                    html.Th("N", style={"textAlign": "right"}),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    striped=True,
                    bordered=True,
                    size="sm",
                ),
            ],
            className="mb-3",
        )

    sections.append(
        dbc.Row(
            [
                dbc.Col(
                    make_interaction_table(
                        synergies, "Top Synergies (I > 0)", "green"
                    ),
                    md=6,
                ),
                dbc.Col(
                    make_interaction_table(
                        redundancies, "Top Redundancies (I < 0)", "red"
                    ),
                    md=6,
                ),
            ]
        )
    )

    # Build heatmap if we have enough data
    if len(results.pairwise) >= 3:
        # Create token index
        all_tokens = set()
        for s in results.pairwise:
            all_tokens.update(s.tokens)
        token_list = sorted(all_tokens)
        token_to_idx = {t: i for i, t in enumerate(token_list)}
        n = len(token_list)

        # Fill matrix
        matrix = np.zeros((n, n))
        for s in results.pairwise:
            i = token_to_idx[s.tokens[0]]
            j = token_to_idx[s.tokens[1]]
            matrix[i, j] = s.interaction_value
            matrix[j, i] = s.interaction_value

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=token_list,
                y=token_list,
                colorscale=[
                    [0, "blue"],
                    [0.5, "white"],
                    [1, "red"],
                ],
                zmid=0,
                colorbar=dict(title="Interaction"),
            )
        )
        fig.update_layout(
            title="Pairwise Interaction Heatmap",
            xaxis_title="Token",
            yaxis_title="Token",
            height=max(400, 50 * n),
            xaxis=dict(tickangle=45),
        )

        sections.append(dcc.Graph(figure=fig))

    # 3-way interactions if available
    if results.three_way:
        rows = []
        for s in results.three_way[:10]:
            tokens_str = " + ".join(s.tokens)
            rows.append(
                html.Tr(
                    [
                        html.Td(tokens_str),
                        html.Td(
                            f"{s.interaction_value:+.4f}",
                            style={
                                "textAlign": "right",
                                "color": (
                                    "green"
                                    if s.interaction_value > 0
                                    else "red"
                                ),
                            },
                        ),
                        html.Td(
                            f"±{s.std_error:.4f}", style={"textAlign": "right"}
                        ),
                    ]
                )
            )

        sections.append(
            html.Div(
                [
                    html.H6("3-Way Interactions"),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Token Triad"),
                                        html.Th(
                                            "I", style={"textAlign": "right"}
                                        ),
                                        html.Th(
                                            "SE", style={"textAlign": "right"}
                                        ),
                                    ]
                                )
                            ),
                            html.Tbody(rows),
                        ],
                        striped=True,
                        bordered=True,
                        size="sm",
                    ),
                ],
                className="mt-4",
            )
        )

    return html.Div(sections)


def build_minimal_sets_display(results: MinimalSetResults) -> html.Div:
    """Build display for minimal activating sets results."""
    if results is None:
        return html.Div("No results to display.", className="text-muted")

    sections = []

    # Summary
    avg_reduction = sum(
        1 - len(ms.minimal_tokens) / len(ms.original_tokens)
        for ms in results.minimal_sets
        if len(ms.original_tokens) > 0
    ) / len(results.minimal_sets)

    sections.append(
        dbc.Alert(
            [
                html.Strong("Minimal Sets Analysis: "),
                f"{results.n_examples_analyzed} examples analyzed, "
                f"threshold = {results.threshold_ratio:.0%} retention, "
                f"avg token reduction = {avg_reduction:.1%}",
            ],
            color="info",
            className="mb-3",
        )
    )

    # Common cores
    if results.common_cores:
        core_rows = []
        for core, count in results.common_cores[:15]:
            core_str = ", ".join(core)
            core_rows.append(
                html.Tr(
                    [
                        html.Td(core_str),
                        html.Td(str(len(core)), style={"textAlign": "center"}),
                        html.Td(str(count), style={"textAlign": "right"}),
                    ]
                )
            )

        sections.append(
            html.Div(
                [
                    html.H6(
                        "Common Cores (recurring patterns in minimal sets)"
                    ),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Core Tokens"),
                                        html.Th(
                                            "Size",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Th(
                                            "Count",
                                            style={"textAlign": "right"},
                                        ),
                                    ]
                                )
                            ),
                            html.Tbody(core_rows),
                        ],
                        striped=True,
                        bordered=True,
                        size="sm",
                    ),
                ],
                className="mb-4",
            )
        )

    # Individual minimal sets
    rows = []
    for ms in results.minimal_sets[:20]:
        orig_str = ", ".join(ms.original_tokens[:6])
        if len(ms.original_tokens) > 6:
            orig_str += f" (+{len(ms.original_tokens) - 6} more)"
        min_str = ", ".join(ms.minimal_tokens)

        rows.append(
            html.Tr(
                [
                    html.Td(min_str, style={"fontWeight": "bold"}),
                    html.Td(orig_str, className="text-muted small"),
                    html.Td(
                        f"{len(ms.original_tokens)} → {len(ms.minimal_tokens)}",
                        style={"textAlign": "center"},
                    ),
                    html.Td(
                        f"{ms.retention_ratio:.1%}",
                        style={"textAlign": "right"},
                    ),
                ]
            )
        )

    sections.append(
        html.Div(
            [
                html.H6("Individual Minimal Sets"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Minimal Set"),
                                    html.Th("Original Tokens"),
                                    html.Th(
                                        "Reduction",
                                        style={"textAlign": "center"},
                                    ),
                                    html.Th(
                                        "Retention",
                                        style={"textAlign": "right"},
                                    ),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                ),
            ]
        )
    )

    return html.Div(sections)


# ---------------------------------------------------------------------------
# Markdown Formatters for Clipboard Export
# ---------------------------------------------------------------------------


def format_itemsets_as_markdown(results: ContrastiveItemsetResults) -> str:
    """Format itemset results as Markdown for clipboard export."""
    if results is None:
        return ""

    lines = [
        "## Contrastive Frequent Itemsets",
        "",
        f"**Feature:** {results.feature_id}",
        f"**High activation:** Top {results.high_threshold_pct}% "
        f"({results.n_high_examples} examples)",
        f"**Baseline:** 25-75% ({results.n_baseline_examples} examples)",
        "",
    ]

    for size in [2, 3, 4]:
        itemsets = results.itemsets_by_size.get(size, [])
        if not itemsets:
            continue

        lines.append(f"### Size-{size} Itemsets")
        lines.append("")
        lines.append(
            "| Tokens | Lift | Support (High) | Support (Base) | Count |"
        )
        lines.append(
            "|--------|------|----------------|----------------|-------|"
        )

        for item in itemsets[:20]:
            tokens_str = ", ".join(item.tokens)
            lines.append(
                f"| {tokens_str} | {item.lift:.2f} | "
                f"{item.support_high:.1%} | {item.support_baseline:.1%} | "
                f"{item.count_high} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_interactions_as_markdown(results: InteractionSweepResults) -> str:
    """Format interaction sweep results as Markdown for clipboard export."""
    if results is None or len(results.pairwise) == 0:
        return ""

    lines = [
        "## Interaction Sweep Results",
        "",
        f"**Feature:** {results.feature_id}",
        f"**Tokens tested:** {len(results.candidate_tokens)}",
        f"**Base contexts:** {results.n_base_contexts}",
        f"**Pairs computed:** {len(results.pairwise)}",
        "",
    ]

    # Top synergies
    synergies = [s for s in results.pairwise if s.interaction_value > 0][:15]
    if synergies:
        lines.append("### Top Synergies (I > 0)")
        lines.append("")
        lines.append("| Token Pair | I | SE | N |")
        lines.append("|------------|---|----|----|")
        for s in synergies:
            tokens_str = " + ".join(s.tokens)
            lines.append(
                f"| {tokens_str} | {s.interaction_value:+.4f} | "
                f"±{s.std_error:.4f} | {s.n_contexts} |"
            )
        lines.append("")

    # Top redundancies
    redundancies = [s for s in results.pairwise if s.interaction_value < 0][:15]
    if redundancies:
        lines.append("### Top Redundancies (I < 0)")
        lines.append("")
        lines.append("| Token Pair | I | SE | N |")
        lines.append("|------------|---|----|----|")
        for s in redundancies:
            tokens_str = " + ".join(s.tokens)
            lines.append(
                f"| {tokens_str} | {s.interaction_value:+.4f} | "
                f"±{s.std_error:.4f} | {s.n_contexts} |"
            )
        lines.append("")

    # 3-way interactions
    if results.three_way:
        lines.append("### 3-Way Interactions")
        lines.append("")
        lines.append("| Token Triad | I | SE |")
        lines.append("|-------------|---|----|")
        for s in results.three_way[:10]:
            tokens_str = " + ".join(s.tokens)
            lines.append(
                f"| {tokens_str} | {s.interaction_value:+.4f} | "
                f"±{s.std_error:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_minimal_sets_as_markdown(results: MinimalSetResults) -> str:
    """Format minimal sets results as Markdown for clipboard export."""
    if results is None:
        return ""

    avg_reduction = sum(
        1 - len(ms.minimal_tokens) / len(ms.original_tokens)
        for ms in results.minimal_sets
        if len(ms.original_tokens) > 0
    ) / max(len(results.minimal_sets), 1)

    lines = [
        "## Minimal Activating Sets",
        "",
        f"**Feature:** {results.feature_id}",
        f"**Threshold:** {results.threshold_ratio:.0%} retention",
        f"**Examples analyzed:** {results.n_examples_analyzed}",
        f"**Average token reduction:** {avg_reduction:.1%}",
        "",
    ]

    # Common cores
    if results.common_cores:
        lines.append("### Common Cores")
        lines.append("")
        lines.append("| Core Tokens | Size | Count |")
        lines.append("|-------------|------|-------|")
        for core, count in results.common_cores[:15]:
            core_str = ", ".join(core)
            lines.append(f"| {core_str} | {len(core)} | {count} |")
        lines.append("")

    # Individual minimal sets
    lines.append("### Individual Minimal Sets")
    lines.append("")
    lines.append("| Minimal Set | Original Size | Minimal Size | Retention |")
    lines.append("|-------------|---------------|--------------|-----------|")
    for ms in results.minimal_sets[:20]:
        min_str = ", ".join(ms.minimal_tokens)
        lines.append(
            f"| {min_str} | {len(ms.original_tokens)} | "
            f"{len(ms.minimal_tokens)} | {ms.retention_ratio:.1%} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callbacks for Higher-Order Analysis
# ---------------------------------------------------------------------------


@dash.callback(
    Output("interaction-tokens-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_interaction_tokens_dropdown(_):
    """Populate token dropdown for interaction sweep."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if (
        not hasattr(DASHBOARD_CONTEXT, "vocab")
        or DASHBOARD_CONTEXT.vocab is None
    ):
        return []

    return [
        {"label": tok, "value": tok}
        for tok in sorted(DASHBOARD_CONTEXT.vocab.keys())
        if not tok.startswith("<")
    ]


@dash.callback(
    [
        Output("itemset-results-display", "children"),
        Output("itemset-markdown-content", "children"),
    ],
    Input("run-itemset-mining-button", "n_clicks"),
    [
        State("feature-dropdown", "value"),
        State("itemset-high-threshold", "value"),
        State("itemset-family-mode", "value"),
        State("active-tab-store", "data"),
    ],
    prevent_initial_call=True,
)
def run_itemset_mining_callback(
    n_clicks, feature_id, threshold, family_mode, active_tab
):
    """Run contrastive frequent itemset mining."""
    if active_tab != "tab-ablation":
        return dash.no_update, dash.no_update

    if feature_id is None:
        return dbc.Alert("Please select a feature first.", color="warning"), ""

    try:
        family = "family" in (family_mode or [])
        results = compute_contrastive_itemsets(
            feature_id=feature_id,
            high_pct=threshold or 5.0,
            family_mode=family,
        )
        return build_itemset_display(results), format_itemsets_as_markdown(
            results
        )
    except Exception as e:
        logger.exception("Error in itemset mining")
        return dbc.Alert(f"Error: {str(e)}", color="danger"), ""


@dash.callback(
    [
        Output("interaction-results-display", "children"),
        Output("interaction-markdown-content", "children"),
    ],
    Input("run-interaction-sweep-button", "n_clicks"),
    [
        State("feature-dropdown", "value"),
        State("interaction-tokens-dropdown", "value"),
        State("interaction-n-contexts", "value"),
        State("interaction-options", "value"),
        State("active-tab-store", "data"),
    ],
    prevent_initial_call=True,
)
def run_interaction_sweep_callback(
    n_clicks, feature_id, tokens, n_contexts, options, active_tab
):
    """Run interaction sweep analysis."""
    if active_tab != "tab-ablation":
        return dash.no_update, dash.no_update

    if feature_id is None:
        return dbc.Alert("Please select a feature first.", color="warning"), ""

    if not tokens or len(tokens) < 2:
        return (
            dbc.Alert(
                "Please select at least 2 tokens to analyze.", color="warning"
            ),
            "",
        )

    if len(tokens) > 50:
        return (
            dbc.Alert(
                "Too many tokens selected. Please select 50 or fewer.",
                color="warning",
            ),
            "",
        )

    try:
        options = options or []
        family_mode = "family" in options
        three_way = "three_way" in options

        results = compute_interaction_sweep(
            feature_id=feature_id,
            candidate_tokens=tokens,
            n_contexts=n_contexts or 50,
            family_mode=family_mode,
            include_three_way=three_way,
        )
        return (
            build_interaction_heatmap(results),
            format_interactions_as_markdown(results),
        )
    except Exception as e:
        logger.exception("Error in interaction sweep")
        return dbc.Alert(f"Error: {str(e)}", color="danger"), ""


@dash.callback(
    [
        Output("minimal-sets-results-display", "children"),
        Output("minimal-sets-markdown-content", "children"),
    ],
    Input("run-minimal-sets-button", "n_clicks"),
    [
        State("feature-dropdown", "value"),
        State("minimal-set-threshold", "value"),
        State("minimal-set-n-examples", "value"),
        State("active-tab-store", "data"),
    ],
    prevent_initial_call=True,
)
def run_minimal_sets_callback(
    n_clicks, feature_id, threshold, n_examples, active_tab
):
    """Find minimal activating sets."""
    if active_tab != "tab-ablation":
        return dash.no_update, dash.no_update

    if feature_id is None:
        return dbc.Alert("Please select a feature first.", color="warning"), ""

    try:
        results = compute_minimal_activating_sets(
            feature_id=feature_id,
            n_examples=n_examples or 50,
            threshold_ratio=threshold or 0.8,
        )
        return (
            build_minimal_sets_display(results),
            format_minimal_sets_as_markdown(results),
        )
    except Exception as e:
        logger.exception("Error in minimal sets analysis")
        return dbc.Alert(f"Error: {str(e)}", color="danger"), ""


# Make the layout accessible for app.py
ablation_component = layout
