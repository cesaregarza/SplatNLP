"""Compute multiplicity vs amplitude axes at the greedy-closure branch point.

This is meant to separate:
  - multiplicity: how many plausible continuations exist (H0 / B_eff)
  - amplitude: how much variability is left (cheap Bernoulli disagreement ÂΔ)

Amplitude approximation (no sampling):
  ÂΔ(S*) = Σ_u 2·p(u|S*)·(1-p(u|S*))

We compute this on:
  - all non-special tokens (omit conditioned tokens)
  - next-tier support (omit conditioned + stack_policy=next-tier)

Inputs:
  - a detailed info_gain sweep JSON for branch contexts + H0/IG
  - (optional) the IG/E[Δdepth] sweep JSON for IG_density join
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch

from splatnlp.mechinterp.analysis.information_gain import (
    allowed_tokens_from_vocab,
    stacking_family_levels,
    support_tokens_for_state,
)
from splatnlp.model.models import SetCompletionModel
from splatnlp.utils.constants import NULL
from splatnlp.utils.infer import build_predict_abilities_batch_multiweapon


def _load_json(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())


def _load_model(
    checkpoint: Path,
    *,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    device: torch.device,
) -> SetCompletionModel:
    model = SetCompletionModel(
        len(vocab),
        len(weapon_vocab),
        32,
        512,
        len(vocab),
        num_layers=3,
        num_heads=8,
        num_inducing_points=32,
        use_layer_norm=True,
        dropout=0.0,
        pad_token_id=vocab["<PAD>"],
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _depth(tokens: list[str]) -> int:
    return int(sum(1 for t in tokens if t != NULL))


def _amp_hat(probs: dict[str, float], support: list[str]) -> float:
    total = 0.0
    for tok in support:
        p = float(probs.get(tok, 0.0))
        total += 2.0 * p * (1.0 - p)
    return float(total)


def _budget(probs: dict[str, float], support: list[str]) -> float:
    return float(sum(float(probs.get(tok, 0.0)) for tok in support))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_report(
    path: Path,
    *,
    payload: dict[str, Any],
    top_n: int,
) -> None:
    meta = payload.get("meta") or {}
    weapons = payload.get("weapons") or []
    stats = payload.get("stats") or {}

    def _fmt(x: float) -> str:
        return f"{float(x):.6f}"

    lines: list[str] = []
    lines.append("# Multiplicity vs Amplitude (Greedy-Closure Branch Point)")
    lines.append("")
    lines.append("This report separates:")
    lines.append("")
    lines.append(
        "- Multiplicity: `H0` and `B_eff = (log_base ** H0)` "
        "(effective # of next-branches)"
    )
    lines.append(
        "- Amplitude (cheap): `ÂΔ = Σ 2·p·(1-p)` (expected token disagreement "
        "under an independent Bernoulli approximation)"
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- IG source: `{meta.get('ig_json')}`")
    lines.append(f"- Checkpoint: `{meta.get('checkpoint')}`")
    lines.append(f"- Device: `{meta.get('device')}`")
    lines.append(f"- Greedy threshold: `{meta.get('greedy_threshold')}`")
    lines.append(f"- Stack policy: `{meta.get('stack_policy')}`")
    lines.append(f"- Omit conditioned: `{meta.get('omit_conditioned')}`")
    lines.append(f"- Log base: `{meta.get('log_base')}`")
    if meta.get("ig_depth_delta_json"):
        lines.append(
            f"- IG/E[Δdepth] source: `{meta.get('ig_depth_delta_json')}`"
        )
    lines.append("")

    if stats:
        lines.append("## Distributions (Per Weapon)")
        lines.append("")
        for key in [
            "h0",
            "b_eff",
            "amp_hat_next_tier",
            "amp_hat_all",
            "expected_delta_depth",
            "ig_density",
        ]:
            s = stats.get(key)
            if not isinstance(s, dict) or not s:
                continue
            lines.append(
                f"- `{key}`: mean {_fmt(s['mean'])}, "
                f"median {_fmt(s['median'])}, min {_fmt(s['min'])}, "
                f"max {_fmt(s['max'])}"
            )
        lines.append("")

    def _top_table(key: str, title: str) -> None:
        rows = sorted(
            weapons, key=lambda w: float(w.get(key, 0.0)), reverse=True
        )
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| rank | weapon_id | weapon_label | depth | H0 | B_eff | "
            "ÂΔ(next-tier) | IG | IG_density |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for i, row in enumerate(rows[:top_n], start=1):
            ig_density = row.get("ig_density")
            lines.append(
                f"| {i} | `{row['weapon_id']}` | {row['weapon_label']} | "
                f"{int(row['depth_before'])} | {_fmt(row['h0'])} | "
                f"{_fmt(row['b_eff'])} | {_fmt(row['amp_hat_next_tier'])} | "
                f"{_fmt(row['information_gain'])} | "
                f"{_fmt(ig_density) if ig_density is not None else ''} |"
            )
        lines.append("")

    _top_table("b_eff", f"Top {top_n} By Multiplicity (`B_eff`)")
    _top_table(
        "amp_hat_next_tier", f"Top {top_n} By Amplitude (`ÂΔ`, next-tier)"
    )

    # Quadrants: median split on B_eff and ÂΔ(next-tier).
    b_vals = np.array([float(w["b_eff"]) for w in weapons], dtype=np.float64)
    a_vals = np.array(
        [float(w["amp_hat_next_tier"]) for w in weapons], dtype=np.float64
    )
    b_med = float(np.median(b_vals)) if len(b_vals) else 0.0
    a_med = float(np.median(a_vals)) if len(a_vals) else 0.0

    def _quad(w: dict[str, Any]) -> str:
        high_b = float(w["b_eff"]) >= b_med
        high_a = float(w["amp_hat_next_tier"]) >= a_med
        if high_b and high_a:
            return "high_mult__high_amp"
        if high_b and not high_a:
            return "high_mult__low_amp"
        if (not high_b) and high_a:
            return "low_mult__high_amp"
        return "low_mult__low_amp"

    groups: dict[str, list[dict[str, Any]]] = {
        "high_mult__high_amp": [],
        "high_mult__low_amp": [],
        "low_mult__high_amp": [],
        "low_mult__low_amp": [],
    }
    for w in weapons:
        groups[_quad(w)].append(w)

    lines.append("## Quadrants (Median Split)")
    lines.append("")
    lines.append(f"- `B_eff` median: {_fmt(b_med)}")
    lines.append(f"- `ÂΔ(next-tier)` median: {_fmt(a_med)}")
    lines.append("")

    def _quad_block(key: str, title: str) -> None:
        rows = groups[key]
        rows.sort(
            key=lambda w: float(w["b_eff"]) * float(w["amp_hat_next_tier"]),
            reverse=True,
        )
        lines.append(f"### {title} (n={len(rows)})")
        lines.append("")
        lines.append(
            "| rank | weapon_id | weapon_label | B_eff | ÂΔ(next-tier) |"
        )
        lines.append("|---:|---|---|---:|---:|")
        for i, row in enumerate(rows[: min(10, len(rows))], start=1):
            lines.append(
                f"| {i} | `{row['weapon_id']}` | {row['weapon_label']} | "
                f"{_fmt(row['b_eff'])} | {_fmt(row['amp_hat_next_tier'])} |"
            )
        lines.append("")

    _quad_block("high_mult__high_amp", "High Multiplicity + High Amplitude")
    _quad_block(
        "high_mult__low_amp", "High Multiplicity + Low Amplitude (flavors)"
    )
    _quad_block(
        "low_mult__high_amp",
        "Low Multiplicity + High Amplitude (few big modes)",
    )
    _quad_block(
        "low_mult__low_amp", "Low Multiplicity + Low Amplitude (unimodal-ish)"
    )

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ig-json",
        type=Path,
        default=Path(
            "tmp_results/"
            "info_gain_all_weapons_ultra_greedy0.7_"
            "omit_conditioned_next_tier_detailed.json"
        ),
        help="Detailed info_gain_cli --all-weapons JSON (must include branch_context).",
    )
    parser.add_argument(
        "--ig-depth-delta-json",
        type=Path,
        default=Path(
            "tmp_results/ig_per_expected_delta_depth_all_weapons_ultra_"
            "greedy0.7_next_tier.json"
        ),
        help="Optional IG/E[Δdepth] JSON to join (for IG_density).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saved_models/dataset_v0_2_super/clean_slate.pth"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("saved_models/dataset_v0_2_full/vocab.json"),
    )
    parser.add_argument(
        "--weapon-vocab",
        type=Path,
        default=Path("saved_models/dataset_v0_2_full/weapon_vocab.json"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--stack-policy",
        type=str,
        default="next-tier",
        choices=["none", "next-tier", "max-tier"],
    )
    parser.add_argument(
        "--include-conditioned",
        action="store_true",
        help="Include conditioned tokens in amplitude supports.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many weapons per top table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to tmp_results/...json).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional markdown report output path.",
    )
    args = parser.parse_args()

    ig_payload = orjson.loads(args.ig_json.read_bytes())
    ig_meta = ig_payload.get("meta") or {}
    weapons_in = ig_payload.get("weapons") or []
    if not isinstance(weapons_in, list):
        raise ValueError("Unexpected IG JSON shape (weapons is not a list)")

    log_base = float(ig_meta.get("log_base", 2.0))
    greedy_threshold = float(ig_meta.get("greedy_threshold", 0.7))

    join_density: dict[str, dict[str, float]] = {}
    if (
        args.ig_depth_delta_json is not None
        and args.ig_depth_delta_json.exists()
    ):
        d_payload = orjson.loads(args.ig_depth_delta_json.read_bytes())
        d_weapons = d_payload.get("weapons") or []
        if isinstance(d_weapons, list):
            for row in d_weapons:
                wid = str(row.get("weapon_id"))
                join_density[wid] = {
                    "expected_delta_depth": float(
                        row.get("expected_delta_depth", 0.0)
                    ),
                    "ig_density": (
                        None
                        if row.get("ig_per_expected_delta_depth") is None
                        else float(row.get("ig_per_expected_delta_depth"))
                    ),
                }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")

    vocab = _load_json(args.vocab)
    weapon_vocab = _load_json(args.weapon_vocab)
    model = _load_model(
        args.checkpoint, vocab=vocab, weapon_vocab=weapon_vocab, device=device
    )

    allowed_tokens = allowed_tokens_from_vocab(vocab)
    levels_by_family = stacking_family_levels(allowed_tokens)
    omit_conditioned = not bool(args.include_conditioned)

    contexts: list[list[str]] = []
    weapon_ids: list[str] = []
    rows_base: list[dict[str, Any]] = []
    for row in weapons_in:
        wid = str(row.get("weapon_id"))
        branch_context = list(row.get("branch_context") or [])
        contexts.append(branch_context)
        weapon_ids.append(wid)
        rows_base.append(row)

    predict_batch_multiweapon = build_predict_abilities_batch_multiweapon(
        vocab,
        weapon_vocab,
        pad_token="<PAD>",
        hook=None,
        device=device,
        output_type="dict",
    )

    probs_out: list[dict[str, float]] = []
    for start in range(0, len(contexts), int(args.batch_size)):
        chunk_ctx = contexts[start : start + int(args.batch_size)]
        chunk_wids = weapon_ids[start : start + int(args.batch_size)]
        probs_out.extend(
            predict_batch_multiweapon(model, chunk_ctx, chunk_wids)
        )

    weapons_out: list[dict[str, Any]] = []
    h0s: list[float] = []
    beffs: list[float] = []
    a_next: list[float] = []
    a_all: list[float] = []
    exp_deltas: list[float] = []
    ig_densities: list[float] = []

    for base, probs in zip(rows_base, probs_out):
        wid = str(base.get("weapon_id"))
        label = str(base.get("weapon_label", ""))
        branch_context = list(base.get("branch_context") or [])
        conditioned = set(branch_context)
        conditioned.discard(NULL)

        support_all = support_tokens_for_state(
            allowed_tokens=list(allowed_tokens),
            conditioned_tokens=conditioned,
            omit_conditioned=omit_conditioned,
            stack_policy="none",
            levels_by_family=levels_by_family,
        )
        support_next = support_tokens_for_state(
            allowed_tokens=list(allowed_tokens),
            conditioned_tokens=conditioned,
            omit_conditioned=omit_conditioned,
            stack_policy=str(args.stack_policy),
            levels_by_family=levels_by_family,
        )

        h0 = float(base.get("h0", 0.0))
        b_eff = float(log_base**h0) if h0 > 0 else 0.0
        ig = float(base.get("information_gain", 0.0))

        amp_all = _amp_hat(probs, support_all)
        amp_next = _amp_hat(probs, support_next)

        out_row: dict[str, Any] = {
            "weapon_id": wid,
            "weapon_label": label,
            "depth_before": _depth(branch_context),
            "branch_context": branch_context,
            "h0": h0,
            "b_eff": b_eff,
            "information_gain": ig,
            "amp_hat_all": amp_all,
            "budget_all": _budget(probs, support_all),
            "support_all_len": int(len(support_all)),
            "amp_hat_next_tier": amp_next,
            "budget_next_tier": _budget(probs, support_next),
            "support_next_tier_len": int(len(support_next)),
        }

        joined = join_density.get(wid)
        if joined is not None:
            out_row.update(joined)
            if joined.get("expected_delta_depth") is not None:
                exp_deltas.append(float(joined["expected_delta_depth"]))
            if joined.get("ig_density") is not None:
                ig_densities.append(float(joined["ig_density"]))

        weapons_out.append(out_row)
        h0s.append(h0)
        beffs.append(b_eff)
        a_next.append(amp_next)
        a_all.append(amp_all)

    out_path = (
        args.out
        if args.out is not None
        else Path("tmp_results")
        / (
            "multimodality_axes_ultra_"
            f"greedy{format(float(greedy_threshold), 'g')}_"
            f"{str(args.stack_policy)}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": {
            "ig_json": str(args.ig_json),
            "ig_depth_delta_json": (
                None
                if args.ig_depth_delta_json is None
                else str(args.ig_depth_delta_json)
            ),
            "checkpoint": str(args.checkpoint),
            "device": str(args.device),
            "greedy_threshold": float(greedy_threshold),
            "stack_policy": str(args.stack_policy),
            "omit_conditioned": bool(omit_conditioned),
            "log_base": float(log_base),
            "out_path": str(out_path),
        },
        "weapons": weapons_out,
        "stats": {
            "h0": _summary(h0s),
            "b_eff": _summary(beffs),
            "amp_hat_next_tier": _summary(a_next),
            "amp_hat_all": _summary(a_all),
            "expected_delta_depth": _summary(exp_deltas),
            "ig_density": _summary(ig_densities),
        },
    }
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    print(f"Wrote {out_path}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(args.report, payload=payload, top_n=int(args.top_n))
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
