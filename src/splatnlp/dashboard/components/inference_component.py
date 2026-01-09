"""Inference component for running model predictions on partial builds."""

import logging
from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, callback, ctx, dcc, html

from splatnlp.utils.constants import MAIN_ONLY_ABILITIES, STANDARD_ABILITIES

logger = logging.getLogger(__name__)

ALL_ABILITIES = sorted(set(MAIN_ONLY_ABILITIES + STANDARD_ABILITIES))
ABILITY_OPTIONS = [
    {"label": a.replace("_", " ").title(), "value": a} for a in ALL_ABILITIES
]

MAIN_ONLY_SET = set(MAIN_ONLY_ABILITIES)


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
        # Abilities section
        html.H5("Abilities", className="mt-3"),
        html.P(
            "Add abilities and specify how many main/sub slots they occupy.",
            className="text-muted small",
        ),
        # Store for ability rows data
        dcc.Store(
            id="inference-abilities-store",
            storage_type="memory",
            data=[],
        ),
        # Dynamic ability rows container
        html.Div(id="inference-ability-rows", className="mb-3"),
        # Add ability button
        dbc.Button(
            "+ Add Ability",
            id="add-ability-btn",
            color="secondary",
            size="sm",
            className="mb-3",
        ),
        # AP summary
        html.Div(id="inference-ap-summary", className="mb-3"),
        # Run button
        dbc.Button(
            "Run Inference",
            id="run-inference-btn",
            color="primary",
            size="lg",
            className="mb-4",
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


def render_ability_row(row_data: dict, row_index: int) -> dbc.Card:
    """Render a single ability row."""
    ability = row_data.get("ability")
    mains = row_data.get("mains", 0)
    subs = row_data.get("subs", 0)

    is_main_only = ability in MAIN_ONLY_SET if ability else False

    # Ability dropdown
    ability_dropdown = dcc.Dropdown(
        id={"type": "ability-select", "index": row_index},
        options=ABILITY_OPTIONS,
        value=ability,
        placeholder="Select ability...",
        clearable=True,
        style={"minWidth": "200px"},
    )

    # Slot selectors (only for standard abilities)
    if is_main_only:
        slot_selectors = html.Span(
            "(Main Only - 1 main)",
            className="text-muted ms-3 align-self-center",
        )
    else:
        slot_selectors = html.Div(
            [
                html.Label("Mains:", className="me-1 ms-3"),
                dcc.Dropdown(
                    id={"type": "mains-select", "index": row_index},
                    options=[{"label": str(i), "value": i} for i in range(4)],
                    value=mains,
                    clearable=False,
                    style={"width": "70px"},
                ),
                html.Label("Subs:", className="me-1 ms-2"),
                dcc.Dropdown(
                    id={"type": "subs-select", "index": row_index},
                    options=[{"label": str(i), "value": i} for i in range(10)],
                    value=subs,
                    clearable=False,
                    style={"width": "70px"},
                ),
            ],
            className="d-flex align-items-center",
        )

    # Remove button
    remove_btn = dbc.Button(
        "×",
        id={"type": "remove-ability-btn", "index": row_index},
        color="danger",
        size="sm",
        className="ms-auto",
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        ability_dropdown,
                        slot_selectors,
                        remove_btn,
                    ],
                    className="d-flex align-items-center",
                ),
            ],
            className="py-2",
        ),
        className="mb-2",
    )


def abilities_to_ap_dict(abilities_data: list[dict]) -> dict[str, int]:
    """Convert ability rows to an AP dictionary.

    Main slots = 10 AP each, Sub slots = 3 AP each.
    Main-only abilities always count as 1 main (10 AP).
    """
    ap_dict: dict[str, int] = defaultdict(int)

    for row in abilities_data:
        ability = row.get("ability")
        if not ability:
            continue

        if ability in MAIN_ONLY_SET:
            # Main-only abilities count as 1 main
            ap_dict[ability] += 10
        else:
            # Standard abilities: mains * 10 + subs * 3
            mains = row.get("mains", 0)
            subs = row.get("subs", 0)
            ap_dict[ability] += mains * 10 + subs * 3

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

    inv_weapon_vocab = getattr(DASHBOARD_CONTEXT, "inv_weapon_vocab", None)
    if inv_weapon_vocab is None:
        return []

    _, id_to_name, _ = generate_maps()

    options = []
    for weapon_idx, weapon_key in inv_weapon_vocab.items():
        wid_str = weapon_key.replace("weapon_id_", "")
        weapon_name = id_to_name.get(wid_str, weapon_key)
        if "Replica" not in weapon_name:
            label = f"{weapon_name} ({wid_str})"
            options.append({"label": label, "value": weapon_key})

    options.sort(key=lambda x: x["label"])
    return options


# Callback to manage ability rows (add/remove/update)
@callback(
    Output("inference-abilities-store", "data"),
    [
        Input("add-ability-btn", "n_clicks"),
        Input({"type": "remove-ability-btn", "index": ALL}, "n_clicks"),
        Input({"type": "ability-select", "index": ALL}, "value"),
        Input({"type": "mains-select", "index": ALL}, "value"),
        Input({"type": "subs-select", "index": ALL}, "value"),
    ],
    State("inference-abilities-store", "data"),
    prevent_initial_call=True,
)
def manage_ability_rows(
    add_clicks,
    remove_clicks,
    ability_values,
    mains_values,
    subs_values,
    current_data,
):
    """Manage the ability rows store."""
    if current_data is None:
        current_data = []

    triggered_id = ctx.triggered_id

    # Add new row
    if triggered_id == "add-ability-btn":
        current_data.append({"ability": None, "mains": 0, "subs": 0})
        return current_data

    # Remove row
    if (
        isinstance(triggered_id, dict)
        and triggered_id.get("type") == "remove-ability-btn"
    ):
        remove_index = triggered_id["index"]
        current_data = [
            row for i, row in enumerate(current_data) if i != remove_index
        ]
        # Re-index remaining rows
        return current_data

    # Update ability selection
    if (
        isinstance(triggered_id, dict)
        and triggered_id.get("type") == "ability-select"
    ):
        update_index = triggered_id["index"]
        if update_index < len(current_data):
            new_ability = (
                ability_values[update_index]
                if update_index < len(ability_values)
                else None
            )
            current_data[update_index]["ability"] = new_ability
            # Reset mains/subs if switching to main-only
            if new_ability in MAIN_ONLY_SET:
                current_data[update_index]["mains"] = 0
                current_data[update_index]["subs"] = 0
        return current_data

    # Update mains selection
    if (
        isinstance(triggered_id, dict)
        and triggered_id.get("type") == "mains-select"
    ):
        update_index = triggered_id["index"]
        if update_index < len(current_data):
            current_data[update_index]["mains"] = (
                mains_values[update_index]
                if update_index < len(mains_values)
                else 0
            )
        return current_data

    # Update subs selection
    if (
        isinstance(triggered_id, dict)
        and triggered_id.get("type") == "subs-select"
    ):
        update_index = triggered_id["index"]
        if update_index < len(current_data):
            current_data[update_index]["subs"] = (
                subs_values[update_index]
                if update_index < len(subs_values)
                else 0
            )
        return current_data

    return current_data


# Callback to render ability rows from store
@callback(
    [
        Output("inference-ability-rows", "children"),
        Output("inference-ap-summary", "children"),
    ],
    Input("inference-abilities-store", "data"),
)
def render_ability_rows(abilities_data):
    """Render the ability rows and AP summary from store data."""
    if not abilities_data:
        return (
            html.Div(
                "No abilities added. Click '+ Add Ability' to start.",
                className="text-muted",
            ),
            html.Div(),
        )

    rows = [render_ability_row(row, i) for i, row in enumerate(abilities_data)]

    # Calculate AP summary
    ap_dict = abilities_to_ap_dict(abilities_data)
    total_ap = sum(ap_dict.values())

    if ap_dict:
        ap_parts = [
            f"{a.replace('_', ' ').title()}: {ap}AP"
            for a, ap in sorted(ap_dict.items())
        ]
        summary = dbc.Alert(
            [
                html.Strong(f"Total: {total_ap} AP"),
                html.Br(),
                html.Small(" | ".join(ap_parts)),
            ],
            color="info",
            className="py-2",
        )
    else:
        summary = html.Div()

    return rows, summary


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
        State("inference-abilities-store", "data"),
        State("feature-dropdown", "value"),
        State("inference-build-history", "data"),
    ],
    prevent_initial_call=True,
)
def run_inference(
    n_clicks,
    weapon_id,
    abilities_data,
    current_feature_id,
    build_history,
):
    """Run model inference on the specified partial build."""
    import torch

    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    from splatnlp.serve.tokenize import tokenize_build
    from splatnlp.utils.infer import (
        build_predict_abilities,
        build_predict_abilities_batch,
    )
    from splatnlp.utils.reconstruct.allocator import Allocator
    from splatnlp.utils.reconstruct.beam_search import reconstruct_build

    if build_history is None:
        build_history = []

    def render_history(history):
        if not history:
            return html.Div("No builds yet", className="text-muted")
        return html.Div(
            [format_build_history_entry(e, i) for i, e in enumerate(history)]
        )

    if not weapon_id:
        return (
            dbc.Alert("Please select a weapon.", color="warning"),
            "",
            "",
            "",
            build_history,
            render_history(build_history),
        )

    model = getattr(DASHBOARD_CONTEXT, "primary_model", None)
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
    device = next(model.parameters()).device

    try:
        abilities_data = abilities_data or []
        ap_dict = abilities_to_ap_dict(abilities_data)

        tokens = tokenize_build(ap_dict)

        # Build prediction function (without hook for reconstruction)
        predict_fn_factory = build_predict_abilities(
            vocab=vocab,
            weapon_vocab=weapon_vocab,
            pad_token="<PAD>",
            hook=None,
            device=device,
            output_type="dict",
        )
        predict_batch_fn_factory = build_predict_abilities_batch(
            vocab=vocab,
            weapon_vocab=weapon_vocab,
            pad_token="<PAD>",
            hook=None,
            device=device,
            output_type="dict",
        )

        # Create callable that predict_fn expects
        def predict_fn(current_tokens, wpn_id):
            return predict_fn_factory(model, current_tokens, wpn_id)

        def predict_batch_fn(token_batches, wpn_id):
            return predict_batch_fn_factory(model, list(token_batches), wpn_id)

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
            pd.DataFrame(pred_rows),
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

                # Remove the hook
                handle.remove()
            except Exception as e:
                logger.warning(f"Could not get SAE activation: {e}")

        # Run reconstruction with better parameters
        allocator = Allocator()
        builds = reconstruct_build(
            predict_fn=predict_fn,
            predict_batch_fn=predict_batch_fn,
            weapon_id=weapon_id,
            initial_context=tokens,
            allocator=allocator,
            beam_size=10,
            max_steps=10,
            top_k=1,
        )

        # Format result
        completed_build = None
        if builds:
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

        updated_history = [history_entry] + build_history[:4]

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
