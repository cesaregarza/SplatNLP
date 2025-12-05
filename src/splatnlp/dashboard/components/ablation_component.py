import json  # For potentially displaying dicts or if data is stored as JSON string
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Literal

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import torch  # Import PyTorch
from dash import ALL, Input, Output, State, callback_context, dcc, html

# Import weapon name mapping utility
from splatnlp.dashboard.utils.converters import (
    generate_weapon_name_mapping,
    get_weapon_properties_df,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures for Token Sensitivity Sweep
# ---------------------------------------------------------------------------


@dataclass
class TokenSweepResult:
    """Result of testing a single token."""

    token_name: str
    add_delta: float | None  # Activation change when adding token
    remove_delta: float | None  # Activation change when removing (None if not in base)
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


# ---------------------------------------------------------------------------
# Data Structures for Full Sweep Analysis
# ---------------------------------------------------------------------------

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
    # Individual weapon activations (filtered by threshold) - (name, activation)
    null_base_weapons: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class BuildQueueItem:
    """A build in the queue."""

    weapon_id: int
    weapon_name: str
    ability_tokens: list[str] = field(default_factory=list)
    description: str = ""


# Define the layout for the Ablation tab
layout = html.Div(
    [
        dcc.Store(id="ablation-primary-store"),  # Store for primary build data
        dcc.Store(
            id="ablation-secondary-store"
        ),  # Store for modified/secondary build data
        html.H3("Ablation Analysis"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Primary Build Details"),
                        html.Div(
                            id="primary-build-display",
                            children="No primary build selected.",
                            className="mb-3 p-2 border rounded bg-light",
                            style={"minHeight": "100px"},
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
                                                {"label": "Mean", "value": "mean"},
                                                {"label": "Median", "value": "median"},
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
    State("ablation-primary-store", "data"),
    State("secondary-build-input", "value"),
    State("secondary-weapon-dropdown", "value"),
    State("feature-dropdown", "value"),  # Added state for selected feature
)
def run_ablation_analysis(
    n_clicks,
    primary_data,
    secondary_build_list,
    secondary_weapon_id,
    selected_feature_id,
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

    if not primary_data:
        return "Please select a primary build first."

    if not secondary_build_list or len(secondary_build_list) == 0:
        return "Please select secondary ability tokens."

    # Get primary build info
    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    primary_weapon_id = primary_data.get("weapon_id_token")
    primary_ability_token_ids = primary_data.get("ability_input_tokens", [])

    if primary_weapon_id is None:
        return "Weapon ID token missing from primary build data."

    # Convert primary token IDs back to names
    primary_ability_names = [
        inv_vocab.get(token_id, f"Unknown_{token_id}")
        for token_id in primary_ability_token_ids
    ]

    # Get weapon names (use secondary weapon if selected, otherwise use primary)
    secondary_weapon_id = (
        secondary_weapon_id
        if secondary_weapon_id is not None
        else primary_weapon_id
    )

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
        return "Could not compute primary activation. Check model and inputs."

    # Compute secondary activation with potentially different weapon
    secondary_activation = compute_feature_activation(
        secondary_build_list, secondary_weapon_raw, selected_feature_id
    )

    if secondary_activation is None:
        return "Could not compute secondary activation. Check model and inputs."

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

    # Calculate difference
    diff = secondary_activation - primary_activation

    # Get English weapon names for display
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
    primary_weapon_english = weapon_name_mapping.get(
        primary_weapon_id, primary_weapon_raw
    )
    secondary_weapon_english = weapon_name_mapping.get(
        secondary_weapon_id, secondary_weapon_raw
    )

    # Display results
    results_display = html.Div(
        [
            html.H5(f"Ablation for {feature_name_or_id}:"),
            html.P(
                f"Primary Build: {', '.join(primary_ability_names)} + {primary_weapon_english}"
            ),
            html.P(f"Primary Activation: {primary_activation:.4f}"),
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
            (
                html.P(
                    f"{'⚠️ Weapon changed' if primary_weapon_id != secondary_weapon_id else ''}",
                    style={"font-style": "italic", "color": "blue"},
                )
                if primary_weapon_id != secondary_weapon_id
                else html.Div()
            ),
        ]
    )

    return results_display


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
    Input("ablation-primary-store", "data"),
    Input("feature-dropdown", "value"),
)
def display_sweep_base_build(primary_data, feature_id):
    """Display the base build info in the sweep section."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not primary_data:
        return html.P(
            "Select a primary build from Top Examples to use as the base build.",
            className="text-muted mb-0",
        )

    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    weapon_id = primary_data.get("weapon_id_token", "N/A")
    ability_token_ids = primary_data.get("ability_input_tokens", [])

    ability_names = [
        str(inv_vocab.get(token_id, token_id))
        for token_id in ability_token_ids
    ]
    abilities_str = ", ".join(ability_names) if ability_names else "None"

    # Get weapon name
    weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
    weapon_raw = inv_weapon_vocab.get(weapon_id, f"Weapon {weapon_id}")
    weapon_english = weapon_name_mapping.get(weapon_id, weapon_raw)

    # Get feature name
    feature_name = f"Feature {feature_id}" if feature_id else "No feature selected"
    if (
        feature_id
        and hasattr(DASHBOARD_CONTEXT, "feature_labels_manager")
        and DASHBOARD_CONTEXT.feature_labels_manager
    ):
        feature_name = DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
            feature_id
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
        feature_name = DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
            feature_id
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
        add_delta = (add_activation - base_activation) if add_activation is not None else None

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
        test_weapon_display = weapon_name_mapping.get(weapon_id, test_weapon_raw)

        if is_current:
            # Current weapon has 0 delta
            swap_delta = 0.0
        else:
            # Compute activation with swapped weapon
            swap_activation = compute_feature_activation(
                base_tokens, test_weapon_raw, feature_id
            )
            swap_delta = (swap_activation - base_activation) if swap_activation is not None else None

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
            add_text = f"{result.add_delta:+.4f}" if result.add_delta is not None else "—"
            remove_text = f"{result.remove_delta:+.4f}" if result.remove_delta is not None else "—"
            notes = "(in base)" if result.was_in_base else ""
            lines.append(f"| {result.token_name} | {add_text} | {remove_text} | {notes} |")

        lines.append("")

    # Weapon results
    if sweep_results.weapon_results:
        lines.append("## Weapon Swap Effects")
        lines.append("")
        lines.append("| Weapon | Swap Effect | Notes |")
        lines.append("|--------|-------------|-------|")

        for result in sweep_results.weapon_results:
            delta_text = f"{result.swap_delta:+.4f}" if result.swap_delta is not None else "—"
            notes = "(current)" if result.is_current else ""
            lines.append(f"| {result.weapon_name} | {delta_text} | {notes} |")

        lines.append("")

    return "\n".join(lines)


def build_sweep_results_display(sweep_results: SweepResults) -> html.Div:
    """Build the results display for token and weapon sweeps."""
    if not sweep_results:
        return html.P("No results to display.")

    has_tokens = sweep_results.token_results and len(sweep_results.token_results) > 0
    has_weapons = sweep_results.weapon_results and len(sweep_results.weapon_results) > 0

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
                                    "Copy MD", className="d-block text-muted mt-1"
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
        max_token_delta = max(
            abs(r.add_delta) for r in sweep_results.token_results if r.add_delta is not None
        ) or 1.0

        for result in sweep_results.token_results:
            # Format add delta
            if result.add_delta is not None:
                add_color = "green" if result.add_delta > 0 else "red" if result.add_delta < 0 else "gray"
                add_text = f"{result.add_delta:+.4f}"
                bar_width = min(100, int(abs(result.add_delta) / max_token_delta * 100))
            else:
                add_color = "gray"
                add_text = "—"
                bar_width = 0

            # Format remove delta
            if result.remove_delta is not None:
                remove_color = "green" if result.remove_delta > 0 else "red" if result.remove_delta < 0 else "gray"
                remove_text = f"{result.remove_delta:+.4f}"
            else:
                remove_text = "—"
                remove_color = "gray"

            bar_color = "bg-success" if result.add_delta and result.add_delta > 0 else "bg-danger"

            token_rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                result.token_name,
                                html.Span(
                                    " (in base)",
                                    className="text-muted small",
                                ) if result.was_in_base else "",
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
        max_weapon_delta = max(
            abs(r.swap_delta) for r in sweep_results.weapon_results if r.swap_delta is not None
        ) or 1.0

        for result in sweep_results.weapon_results:
            if result.swap_delta is not None:
                delta_color = "green" if result.swap_delta > 0 else "red" if result.swap_delta < 0 else "gray"
                delta_text = f"{result.swap_delta:+.4f}"
                bar_width = min(100, int(abs(result.swap_delta) / max_weapon_delta * 100))
            else:
                delta_color = "gray"
                delta_text = "—"
                bar_width = 0

            bar_color = "bg-success" if result.swap_delta and result.swap_delta > 0 else "bg-danger"

            weapon_rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                result.weapon_name,
                                html.Span(
                                    " (current)",
                                    className="text-muted small",
                                ) if result.is_current else "",
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
    State("ablation-primary-store", "data"),
    State("sweep-tokens-dropdown", "value"),
    State("sweep-weapons-dropdown", "value"),
    State("feature-dropdown", "value"),
)
def run_sensitivity_sweep(
    n_clicks, primary_data, tokens_to_test, weapons_to_test, feature_id
):
    """Run the token and weapon sensitivity sweep."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if not n_clicks:
        return html.P(
            "Select tokens and/or weapons to test and click 'Run Sweep'.",
            className="text-muted",
        )

    if not primary_data:
        return dbc.Alert(
            "Please select a primary build first (from Top Examples).",
            color="warning",
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

    # Get base build info
    inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

    weapon_id = primary_data.get("weapon_id_token")
    ability_token_ids = primary_data.get("ability_input_tokens", [])

    if weapon_id is None:
        return dbc.Alert("Weapon ID missing from primary build.", color="danger")

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
    ],
    prevent_initial_call=True,
)
def manage_build_queue(
    add_clicks, clear_clicks, remove_clicks, primary_data, current_queue
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
        if not primary_data:
            return current_queue

        inv_vocab = getattr(DASHBOARD_CONTEXT, "inv_vocab", {})
        inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", {})

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
        weapon_name_mapping = generate_weapon_name_mapping(inv_weapon_vocab)
        weapon_raw = inv_weapon_vocab.get(weapon_id, f"Unknown_{weapon_id}")
        weapon_name = weapon_name_mapping.get(weapon_id, weapon_raw)

        # Create description
        abilities_str = ", ".join(ability_tokens[:3])
        if len(ability_tokens) > 3:
            abilities_str += f"... (+{len(ability_tokens) - 3})"
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
        return html.P("No ability tokens in builds.", className="text-muted"), True

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
        feature_name = DASHBOARD_CONTEXT.feature_labels_manager.get_display_name(
            feature_id
        )

    # Compute base activation
    base_activation = compute_feature_activation(base_tokens, weapon_raw, feature_id)
    if base_activation is None:
        return None

    logger.info(f"Build {build_idx + 1}: Base activation = {base_activation:.4f}")

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
            stripped_base, stripped_base_activation = stripped_base_cache[cache_key]
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

            stripped_base_cache[cache_key] = (stripped_base, stripped_base_activation)

        # Compute activation with stripped base + test token
        test_tokens = stripped_base + [token]
        test_activation = compute_feature_activation(test_tokens, weapon_raw, feature_id)

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
            main_only_results.append(MainOnlyResult(ability_name=token, delta=delta))

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

    weapon_deltas: dict[int, tuple[str, float]] = {}  # weapon_id -> (name, delta)

    for i, test_weapon_id in enumerate(all_weapon_ids):
        if (i + 1) % 30 == 0:
            logger.info(f"  Testing weapon {i + 1}/{len(all_weapon_ids)}")

        test_weapon_raw = inv_weapon_vocab.get(
            test_weapon_id, f"Unknown_{test_weapon_id}"
        )
        test_weapon_name = weapon_name_mapping.get(test_weapon_id, test_weapon_raw)

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
    list[tuple[str, float]],
]:
    """
    Run NULL base weapon sweep once (independent of build abilities).

    Args:
        activation_threshold: Filter out weapons where |activation| < threshold.
                              Set to 0.01 * mean_base_activation to keep only
                              weapons with >= 1% of the base activation.

    Returns:
        (null_base_activation, by_special, by_sub, by_class, individual_weapons)
        where individual_weapons is list of (weapon_name, activation) tuples
    """
    logger.info(f"Running NULL base weapon sweep for {len(all_weapon_ids)} weapons...")
    logger.info(f"  Activation threshold: {activation_threshold:.4f}")

    # Compute raw activation for each weapon with NULL base
    # Store raw activations (not deltas) for filtering
    null_weapon_activations: dict[int, tuple[str, float]] = {}

    for test_weapon_id in all_weapon_ids:
        test_weapon_raw = inv_weapon_vocab.get(
            test_weapon_id, f"Unknown_{test_weapon_id}"
        )
        test_weapon_name = weapon_name_mapping.get(test_weapon_id, test_weapon_raw)

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
    null_by_special: dict[str, list[tuple[str, float]]] = {}
    null_by_sub: dict[str, list[tuple[str, float]]] = {}
    null_by_class: dict[str, list[tuple[str, float]]] = {}

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

    # Build list of individual weapons sorted by activation (descending)
    individual_weapons = [
        (name, activation)
        for name, activation in null_weapon_activations.values()
    ]
    individual_weapons.sort(key=lambda x: x[1], reverse=True)

    logger.info("NULL base weapon sweep complete.")
    return (
        null_base_activation,
        to_group_results(null_by_special),
        to_group_results(null_by_sub),
        to_group_results(null_by_class),
        individual_weapons,
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
    hover_label = "Avg Activation" if "Activation" in y_axis_label else "Avg Delta"

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
        lines.append("| Ability | " + " | ".join(str(ap) for ap in [3, 6, 12, 15, 21, 29, 38, 51, 57]) + " |")
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
            lines.append("| Weapon | Activation |")
            lines.append("|--------|------------|")
            for name, activation in results.null_base_weapons:
                lines.append(f"| {name} | {activation:.4f} |")
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
                                    "Copy MD", className="d-block text-muted mt-1"
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
                    figure=build_weapon_group_bar_chart(results.by_sub, "By Sub Weapon"),
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
                html.Tr([
                    html.Td(name, style={"fontSize": "0.85rem"}),
                    html.Td(
                        f"{activation:.4f}",
                        style={"fontSize": "0.85rem", "textAlign": "right"},
                    ),
                ])
                for name, activation in results.null_base_weapons
            ]

            sections.append(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            dbc.Table(
                                [
                                    html.Thead(
                                        html.Tr([
                                            html.Th("Weapon Kit"),
                                            html.Th("Activation", style={"textAlign": "right"}),
                                        ])
                                    ),
                                    html.Tbody(weapon_rows),
                                ],
                                striped=True,
                                bordered=True,
                                hover=True,
                                size="sm",
                                style={"maxHeight": "400px", "overflowY": "auto"},
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
def run_full_sweep_callback(n_clicks, queue_data, feature_id, aggregation_method):
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
        return dbc.Alert(f"Failed to get weapon properties: {e}", color="danger")

    logger.info(
        f"Starting full sweep: {len(queue_data)} builds, "
        f"{len(all_tokens)} tokens, {len(all_weapon_ids)} weapons"
    )

    # Run sweep for each build
    all_results = []
    for i, build in enumerate(queue_data):
        logger.info(f"Processing build {i + 1}/{len(queue_data)}: {build.get('description', '')}")
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

    logger.info(f"Full sweep complete. Aggregating {len(all_results)} results...")

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


# Make the layout accessible for app.py
ablation_component = layout
