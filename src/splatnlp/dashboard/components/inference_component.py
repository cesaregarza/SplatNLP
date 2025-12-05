"""Inference component for running model predictions on partial builds."""

import logging
from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dcc, html

from splatnlp.utils.constants import MAIN_ONLY_ABILITIES, STANDARD_ABILITIES

logger = logging.getLogger(__name__)

# Build ability options for dropdowns
ALL_ABILITIES = sorted(MAIN_ONLY_ABILITIES + STANDARD_ABILITIES)
ABILITY_OPTIONS = [
    {"label": a.replace("_", " ").title(), "value": a} for a in ALL_ABILITIES
]


def _make_sub_row(gear: str, count: int = 3) -> dbc.Row:
    """Create a row of sub ability dropdowns for a gear piece."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Dropdown(
                        id=f"inference-sub-{gear}-{i}",
                        options=ABILITY_OPTIONS,
                        placeholder="(empty)",
                        clearable=True,
                    )
                ],
                md=4,
            )
            for i in range(count)
        ],
        className="mb-2",
    )


inference_component = html.Div(
    id="inference-content",
    children=[
        html.H4("Build Inference", className="mb-3"),
        # Warning banner
        dbc.Alert(
            "Builds are not verified for game constraints. "
            "Invalid combinations may produce unexpected results.",
            color="warning",
            className="mb-3",
        ),
        # Weapon selector (options populated dynamically)
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Weapon", className="fw-bold"),
                        dcc.Dropdown(
                            id="inference-weapon-dropdown",
                            placeholder="Select weapon...",
                            clearable=True,
                        ),
                    ],
                    md=4,
                ),
            ],
            className="mb-3",
        ),
        # Main slots (3 gear pieces)
        html.H5("Main Abilities", className="mt-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Head"),
                        dcc.Dropdown(
                            id="inference-main-head",
                            options=ABILITY_OPTIONS,
                            placeholder="(empty)",
                            clearable=True,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Clothes"),
                        dcc.Dropdown(
                            id="inference-main-clothes",
                            options=ABILITY_OPTIONS,
                            placeholder="(empty)",
                            clearable=True,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Shoes"),
                        dcc.Dropdown(
                            id="inference-main-shoes",
                            options=ABILITY_OPTIONS,
                            placeholder="(empty)",
                            clearable=True,
                        ),
                    ],
                    md=4,
                ),
            ],
            className="mb-3",
        ),
        # Sub slots (9 total, 3 per gear piece)
        html.H5("Sub Abilities", className="mt-3"),
        html.P("Head Subs", className="text-muted small mb-1"),
        _make_sub_row("head"),
        html.P("Clothes Subs", className="text-muted small mb-1"),
        _make_sub_row("clothes"),
        html.P("Shoes Subs", className="text-muted small mb-1"),
        _make_sub_row("shoes"),
        # Run button
        dbc.Button(
            "Run Inference",
            id="run-inference-btn",
            color="primary",
            size="lg",
            className="mb-4 mt-3",
        ),
        # Store for build history
        dcc.Store(id="inference-build-history", storage_type="memory", data=[]),
        # Results
        dcc.Loading(
            id="loading-inference",
            type="default",
            children=[
                html.Div(id="inference-status"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Input Tokens"),
                                html.Pre(
                                    id="inference-tokens-display",
                                    className="bg-light p-3 border rounded",
                                    style={"minHeight": "100px"},
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.H5("Top Predictions"),
                                html.Div(id="inference-predictions-table"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.H5("Completed Build"),
                                html.Div(id="inference-result-display"),
                            ],
                            md=4,
                        ),
                    ],
                ),
            ],
        ),
        # Build history section
        html.Hr(className="mt-4"),
        html.H5("Build History (Last 5)", className="mt-3"),
        html.Div(id="inference-build-history-display"),
    ],
    className="mb-4",
)


def slots_to_ap_dict(
    main_head: str | None,
    main_clothes: str | None,
    main_shoes: str | None,
    sub_slots: list[str | None],
) -> dict[str, int]:
    """Convert slot selections to an AP dictionary.

    Main slots = 10 AP each, Sub slots = 3 AP each.
    """
    ap_dict: dict[str, int] = defaultdict(int)

    # Main slots (10 AP each)
    for ability in [main_head, main_clothes, main_shoes]:
        if ability:
            ap_dict[ability] += 10

    # Sub slots (3 AP each)
    for ability in sub_slots:
        if ability:
            ap_dict[ability] += 3

    return dict(ap_dict)


def format_build_history_entry(entry: dict, index: int) -> dbc.Card:
    """Format a build history entry as a card."""
    build_data = entry.get("build", {})
    feature_id = entry.get("feature_id")
    feature_activation = entry.get("feature_activation")
    weapon = entry.get("weapon_name", "Unknown")
    input_tokens = entry.get("input_tokens", [])
    top_activations = entry.get("top_activations", [])

    # Build header with feature info
    header_parts = [f"#{index + 1}: {weapon}"]
    if feature_id is not None:
        act_str = f"{feature_activation:.4f}" if feature_activation else "N/A"
        header_parts.append(f"Feature {feature_id}: {act_str}")

    # Format build summary
    build_summary = []
    if build_data:
        mains = build_data.get("mains", {})
        subs = build_data.get("subs", {})
        total_ap = build_data.get("total_ap", 0)

        main_strs = [f"{g}: {a}" for g, a in mains.items() if a]
        if main_strs:
            build_summary.append(f"Mains: {', '.join(main_strs)}")

        sub_strs = [f"{a}×{c}" for a, c in subs.items()]
        if sub_strs:
            build_summary.append(f"Subs: {', '.join(sub_strs)}")

        build_summary.append(f"Total AP: {total_ap}")

    # Format top activations
    top_acts_str = ""
    if top_activations:
        top_acts_str = " | ".join(
            [f"F{fid}: {val:.3f}" for fid, val in top_activations[:5]]
        )

    return dbc.Card(
        [
            dbc.CardHeader(" | ".join(header_parts), className="py-1 small"),
            dbc.CardBody(
                [
                    html.Div(
                        f"Input: {', '.join(input_tokens[:5])}{'...' if len(input_tokens) > 5 else ''}",
                        className="small text-muted",
                    ),
                    html.Div(
                        build_summary[0] if build_summary else "No build",
                        className="small",
                    ),
                    html.Div(
                        build_summary[1] if len(build_summary) > 1 else "",
                        className="small",
                    ),
                    html.Div(
                        build_summary[2] if len(build_summary) > 2 else "",
                        className="small fw-bold",
                    ),
                    html.Div(
                        (
                            f"Top activations: {top_acts_str}"
                            if top_acts_str
                            else ""
                        ),
                        className="small text-muted mt-1",
                    ),
                ],
                className="py-2",
            ),
        ],
        className="mb-2",
    )


def format_build_display(build) -> html.Div:
    """Format a Build object for display."""
    if build is None:
        return html.Div("No valid build found", className="text-muted")

    rows = []

    # Mains
    rows.append(html.H6("Main Abilities"))
    for gear, ability in build.mains.items():
        if ability:
            rows.append(
                html.Div(
                    f"  {gear.title()}: {ability.replace('_', ' ').title()}",
                    className="ms-2",
                )
            )
        else:
            rows.append(
                html.Div(
                    f"  {gear.title()}: (empty)", className="ms-2 text-muted"
                )
            )

    # Subs
    if build.subs:
        rows.append(html.H6("Sub Abilities", className="mt-2"))
        for ability, count in sorted(build.subs.items()):
            rows.append(
                html.Div(
                    f"  {ability.replace('_', ' ').title()}: ×{count}",
                    className="ms-2",
                )
            )

    # Total AP
    rows.append(
        html.Div(
            f"Total AP: {build.total_ap}",
            className="mt-2 fw-bold",
        )
    )

    return html.Div(rows, className="bg-light p-3 border rounded")


# Callback to populate weapon options on load
@callback(
    Output("inference-weapon-dropdown", "options"),
    Input("page-load-trigger", "data"),
)
def populate_weapon_options(trigger):
    """Populate weapon dropdown with available weapons."""
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    from splatnlp.preprocessing.transform.mappings import generate_maps

    logger.info(f"populate_weapon_options called with trigger={trigger}")

    if not hasattr(DASHBOARD_CONTEXT, "inv_weapon_vocab"):
        logger.warning("No inv_weapon_vocab in DASHBOARD_CONTEXT")
        return []

    _, id_to_name, _ = generate_maps()
    inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab

    logger.info(f"Found {len(inv_weapon_vocab)} weapons in vocab")

    options = []
    for weapon_idx, weapon_key in inv_weapon_vocab.items():
        # weapon_key is like "weapon_id_40"
        wid_str = weapon_key.replace("weapon_id_", "")
        weapon_name = id_to_name.get(wid_str, weapon_key)
        if "Replica" not in weapon_name:
            # Format: "Weapon Name (ID)" for searchability
            label = f"{weapon_name} ({wid_str})"
            options.append({"label": label, "value": weapon_key})

    logger.info(f"Returning {len(options)} weapon options")

    # Sort by label
    options.sort(key=lambda x: x["label"])
    return options


@callback(
    [
        Output("inference-status", "children"),
        Output("inference-tokens-display", "children"),
        Output("inference-predictions-table", "children"),
        Output("inference-result-display", "children"),
        Output("inference-build-history", "data"),
        Output("inference-build-history-display", "children"),
    ],
    Input("run-inference-btn", "n_clicks"),
    [
        State("inference-weapon-dropdown", "value"),
        State("inference-main-head", "value"),
        State("inference-main-clothes", "value"),
        State("inference-main-shoes", "value"),
        # Head subs
        State("inference-sub-head-0", "value"),
        State("inference-sub-head-1", "value"),
        State("inference-sub-head-2", "value"),
        # Clothes subs
        State("inference-sub-clothes-0", "value"),
        State("inference-sub-clothes-1", "value"),
        State("inference-sub-clothes-2", "value"),
        # Shoes subs
        State("inference-sub-shoes-0", "value"),
        State("inference-sub-shoes-1", "value"),
        State("inference-sub-shoes-2", "value"),
        # Current feature for activation display
        State("feature-dropdown", "value"),
        # Build history
        State("inference-build-history", "data"),
    ],
    prevent_initial_call=True,
)
def run_inference(
    n_clicks,
    weapon_id,
    main_head,
    main_clothes,
    main_shoes,
    sub_head_0,
    sub_head_1,
    sub_head_2,
    sub_clothes_0,
    sub_clothes_1,
    sub_clothes_2,
    sub_shoes_0,
    sub_shoes_1,
    sub_shoes_2,
    current_feature_id,
    build_history,
):
    """Run model inference on the specified partial build."""
    import torch

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    from splatnlp.serve.tokenize import tokenize_build
    from splatnlp.utils.infer import build_predict_abilities
    from splatnlp.utils.reconstruct.allocator import Allocator
    from splatnlp.utils.reconstruct.beam_search import reconstruct_build

    # Initialize history if None
    if build_history is None:
        build_history = []

    # Helper to render current history
    def render_history(history):
        if not history:
            return html.Div("No builds yet", className="text-muted")
        return html.Div(
            [format_build_history_entry(e, i) for i, e in enumerate(history)]
        )

    # Validate inputs
    if not weapon_id:
        return (
            dbc.Alert("Please select a weapon.", color="warning"),
            "",
            "",
            "",
            build_history,
            render_history(build_history),
        )

    if not hasattr(DASHBOARD_CONTEXT, "primary_model"):
        return (
            dbc.Alert(
                "Model not loaded. Run dashboard with --model-path to enable inference.",
                color="danger",
            ),
            "",
            "",
            "",
            build_history,
            render_history(build_history),
        )

    model = DASHBOARD_CONTEXT.primary_model
    if model is None:
        return (
            dbc.Alert(
                "Model not loaded. Run dashboard with --model-path to enable inference.",
                color="danger",
            ),
            "",
            "",
            "",
            build_history,
            render_history(build_history),
        )

    vocab = DASHBOARD_CONTEXT.vocab
    weapon_vocab = DASHBOARD_CONTEXT.weapon_vocab
    sae_model = getattr(DASHBOARD_CONTEXT, "sae_model", None)

    # Collect sub slots
    sub_slots = [
        sub_head_0,
        sub_head_1,
        sub_head_2,
        sub_clothes_0,
        sub_clothes_1,
        sub_clothes_2,
        sub_shoes_0,
        sub_shoes_1,
        sub_shoes_2,
    ]

    try:
        # Convert slots to AP dict
        ap_dict = slots_to_ap_dict(
            main_head, main_clothes, main_shoes, sub_slots
        )
        logger.info(f"AP dict: {ap_dict}")

        # Tokenize the build
        tokens = tokenize_build(ap_dict)
        logger.info(f"Tokens: {tokens}")

        # Build prediction function (without hook for reconstruction)
        predict_fn_factory = build_predict_abilities(
            vocab=vocab,
            weapon_vocab=weapon_vocab,
            pad_token="<PAD>",
            hook=None,
            output_type="dict",
        )

        # Create callable that predict_fn expects
        def predict_fn(current_tokens, wpn_id):
            return predict_fn_factory(model, current_tokens, wpn_id)

        # Get raw predictions for display
        raw_preds = predict_fn(tokens, weapon_id)

        # Sort predictions by probability
        sorted_preds = sorted(
            raw_preds.items(), key=lambda x: x[1], reverse=True
        )

        # Build predictions table (top 20)
        pred_rows = []
        for tok, prob in sorted_preds[:20]:
            if tok.startswith("<"):
                continue
            pred_rows.append({"Token": tok, "Probability": f"{prob:.4f}"})

        pred_table = dbc.Table.from_dataframe(
            __import__("pandas").DataFrame(pred_rows),
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
        )

        # Get SAE activations for the INPUT build (before reconstruction)
        feature_activation = None
        top_activations = []  # List of (feature_id, activation_value) tuples
        all_activations = None  # Full activation array for logging

        if sae_model is not None:
            try:
                from splatnlp.monosemantic_sae.hooks import register_hooks

                device = next(model.parameters()).device

                # Register hook on the model
                hook, handle = register_hooks(
                    model, sae_model, bypass=False, no_change=True
                )

                # Run forward pass to get activations for the INPUT tokens
                input_tokens = [vocab[t] for t in tokens]
                input_tensor = torch.tensor(
                    input_tokens, device=device
                ).unsqueeze(0)
                weapon_tensor = torch.tensor(
                    [weapon_vocab[weapon_id]], device=device
                ).unsqueeze(0)
                key_padding_mask = (input_tensor == vocab["<PAD>"]).to(device)

                model.eval()
                with torch.no_grad():
                    _ = model(
                        input_tensor,
                        weapon_tensor,
                        key_padding_mask=key_padding_mask,
                    )

                # Get all activations from last_h_post
                if hook.last_h_post is not None:
                    all_activations = hook.last_h_post.squeeze().cpu().numpy()

                    # Get top 10 activations
                    top_indices = np.argsort(all_activations)[-10:][::-1]
                    top_activations = [
                        (int(idx), float(all_activations[idx]))
                        for idx in top_indices
                    ]

                    # Get activation for the current feature if specified
                    if (
                        current_feature_id is not None
                        and current_feature_id < len(all_activations)
                    ):
                        feature_activation = float(
                            all_activations[current_feature_id]
                        )
                        logger.info(
                            f"Feature {current_feature_id} activation on input build: {feature_activation:.4f}"
                        )

                    # Log build and top activations
                    logger.info(f"Build tokens: {tokens}")
                    logger.info(f"Top 10 activations: {top_activations}")

                # Remove the hook
                handle.remove()
            except Exception as e:
                logger.warning(f"Could not get SAE activation: {e}")

        # Run reconstruction with better parameters
        allocator = Allocator()
        builds = reconstruct_build(
            predict_fn=predict_fn,
            weapon_id=weapon_id,
            initial_context=tokens,
            allocator=allocator,
            beam_size=10,  # Increased from 5
            max_steps=10,  # Increased from 6
            top_k=1,
        )

        # Format result
        completed_build = None
        if builds and len(builds) > 0:
            completed_build = builds[0]
            result_display = format_build_display(completed_build)
        else:
            result_display = html.Div(
                "No valid build could be constructed.", className="text-warning"
            )

        # Build status message with feature activation if available
        status_parts = [f"Inference complete. Input: {len(tokens)} tokens"]
        if feature_activation is not None:
            status_parts.append(
                f"Feature {current_feature_id} activation: {feature_activation:.4f}"
            )
        status_msg = " | ".join(status_parts)

        # Get weapon name for history
        from splatnlp.preprocessing.transform.mappings import generate_maps

        _, id_to_name, _ = generate_maps()
        wid_str = weapon_id.replace("weapon_id_", "")
        weapon_name = id_to_name.get(wid_str, weapon_id)

        # Create history entry
        history_entry = {
            "input_tokens": tokens,
            "weapon_id": weapon_id,
            "weapon_name": weapon_name,
            "feature_id": current_feature_id,
            "feature_activation": feature_activation,
            "top_activations": top_activations,
            "build": {
                "mains": completed_build.mains if completed_build else {},
                "subs": dict(completed_build.subs) if completed_build else {},
                "total_ap": completed_build.total_ap if completed_build else 0,
            },
        }

        # Prepend to history and limit to 5
        updated_history = [history_entry] + build_history[:4]

        logger.info(
            f"Added build to history. Total entries: {len(updated_history)}"
        )

        return (
            dbc.Alert(status_msg, color="success"),
            "\n".join(tokens) if tokens else "(empty build)",
            pred_table,
            result_display,
            updated_history,
            render_history(updated_history),
        )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return (
            dbc.Alert(f"Error: {str(e)}", color="danger"),
            "",
            "",
            "",
            build_history,
            render_history(build_history),
        )
