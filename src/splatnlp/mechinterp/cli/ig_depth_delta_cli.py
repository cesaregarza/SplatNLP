"""CLI to compute IG / E[Δdepth] requiring additional greedy-closure compute.

This reads a detailed `info_gain_cli --all-weapons` JSON (for branch contexts,
candidate supports, and q=p0) and then, for each candidate token t, performs an
additional greedy closure until the next branching point to measure Δdepth(t).

Metric:
    ig_per_expected_delta_depth = information_gain / E_t~q[Δdepth(t)]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch

from splatnlp.mechinterp.analysis.depth_delta import (
    expected_depth_delta_for_weapon,
)
from splatnlp.model.models import SetCompletionModel
from splatnlp.utils.infer import build_predict_abilities_batch


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


def _summary_stats(values: list[float]) -> dict[str, float]:
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
    lines.append("# IG / E[Δdepth] (Next Branch Depth)")
    lines.append("")
    lines.append(
        "Computes expected depth increase after conditioning on a candidate"
    )
    lines.append(
        "token and greedy-closing to the next branch point, then reports"
    )
    lines.append("`IG / E[Δdepth]`.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- IG source: `{meta.get('ig_json')}`")
    lines.append(f"- Checkpoint: `{meta.get('checkpoint')}`")
    lines.append(f"- Device: `{meta.get('device')}`")
    lines.append(f"- Greedy threshold: `{meta.get('greedy_threshold')}`")
    lines.append(f"- Top-k tokens stored: `{meta.get('top_k_tokens')}`")
    lines.append("")

    if stats:
        lines.append("## Distributions (Per Weapon)")
        lines.append("")
        for key in ["expected_delta_depth", "ig_per_expected_delta_depth"]:
            s = stats.get(key)
            if not isinstance(s, dict) or not s:
                continue
            lines.append(
                f"- `{key}`: mean {_fmt(s['mean'])}, "
                f"median {_fmt(s['median'])}, min {_fmt(s['min'])}, "
                f"max {_fmt(s['max'])}"
            )
        lines.append("")

    with_ratio = [
        w for w in weapons if w.get("ig_per_expected_delta_depth") is not None
    ]
    with_ratio.sort(
        key=lambda w: float(w["ig_per_expected_delta_depth"]), reverse=True
    )

    lines.append(f"## Top {top_n} Weapons By `ig_per_expected_delta_depth`")
    lines.append("")
    lines.append(
        "| rank | weapon_id | weapon_label | IG | E[Δdepth] | "
        "IG/E[Δdepth] | depth_before |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for i, row in enumerate(with_ratio[:top_n], start=1):
        lines.append(
            f"| {i} | `{row['weapon_id']}` | {row['weapon_label']} | "
            f"{_fmt(row['information_gain'])} | "
            f"{_fmt(row['expected_delta_depth'])} | "
            f"{_fmt(row['ig_per_expected_delta_depth'])} | "
            f"{int(row['depth_before'])} |"
        )
    lines.append("")

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
        help="Detailed info_gain_cli --all-weapons JSON (must include p0).",
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
    parser.add_argument(
        "--greedy-threshold",
        type=float,
        default=None,
        help="Override greedy threshold (defaults to IG JSON meta).",
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=15,
        help="How many per-token Δdepth contributions to store.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many weapons to include in markdown tables.",
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
    tokens = ig_payload.get("tokens") or []
    weapons_in = ig_payload.get("weapons") or []
    if not isinstance(tokens, list) or not isinstance(weapons_in, list):
        raise ValueError("Unexpected IG JSON shape")

    greedy_threshold = (
        float(args.greedy_threshold)
        if args.greedy_threshold is not None
        else float(ig_meta.get("greedy_threshold", 0.7))
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")

    vocab = _load_json(args.vocab)
    weapon_vocab = _load_json(args.weapon_vocab)
    model = _load_model(
        args.checkpoint, vocab=vocab, weapon_vocab=weapon_vocab, device=device
    )

    predict_batch_factory = build_predict_abilities_batch(
        vocab,
        weapon_vocab,
        pad_token="<PAD>",
        hook=None,
        device=device,
        output_type="dict",
    )

    def predict_batch_fn(
        contexts: list[list[str]], weapon_id: str
    ) -> list[dict[str, float]]:
        return predict_batch_factory(model, contexts, weapon_id)

    results: list[dict[str, Any]] = []
    exp_deltas: list[float] = []
    ratios: list[float] = []

    for row in weapons_in:
        wid = str(row.get("weapon_id"))
        label = str(row.get("weapon_label", ""))
        ig = float(row.get("information_gain", 0.0))
        branch_context = list(row.get("branch_context") or [])

        p0_vec = row.get("p0")
        if not isinstance(p0_vec, list):
            raise ValueError(
                "Expected detailed IG JSON with per-weapon p0 vectors "
                "(run info_gain_cli with --detailed)."
            )
        if len(p0_vec) != len(tokens):
            raise ValueError(
                f"p0 length mismatch for {wid}: {len(p0_vec)} vs {len(tokens)}"
            )

        support0: list[str] = []
        p0: dict[str, float] = {}
        for tok, p in zip(tokens, p0_vec):
            p = float(p)
            if p <= 0.0:
                continue
            tok = str(tok)
            support0.append(tok)
            p0[tok] = p

        res = expected_depth_delta_for_weapon(
            predict_batch_fn=predict_batch_fn,
            weapon_id=wid,
            weapon_label=label,
            greedy_threshold=float(greedy_threshold),
            branch_context=branch_context,
            support0=support0,
            p0=p0,
            information_gain=float(ig),
            top_k_tokens=int(args.top_k_tokens),
        )
        out_row = res.to_dict()
        out_row["support0_len"] = int(len(support0))
        results.append(out_row)
        exp_deltas.append(float(res.expected_delta_depth))
        if res.ig_per_expected_delta_depth is not None:
            ratios.append(float(res.ig_per_expected_delta_depth))

    with_ratio = [
        r for r in results if r.get("ig_per_expected_delta_depth") is not None
    ]
    without_ratio = [
        r for r in results if r.get("ig_per_expected_delta_depth") is None
    ]
    with_ratio.sort(
        key=lambda r: float(r["ig_per_expected_delta_depth"]), reverse=True
    )
    results_sorted = with_ratio + without_ratio

    out_path = (
        args.out
        if args.out is not None
        else Path("tmp_results")
        / (
            "ig_per_expected_delta_depth_all_weapons_ultra_"
            f"greedy{format(float(greedy_threshold), 'g')}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_payload: dict[str, Any] = {
        "meta": {
            "ig_json": str(args.ig_json),
            "checkpoint": str(args.checkpoint),
            "device": str(args.device),
            "greedy_threshold": float(greedy_threshold),
            "top_k_tokens": int(args.top_k_tokens),
            "out_path": str(out_path),
            "source_meta": dict(ig_meta) if isinstance(ig_meta, dict) else {},
        },
        "weapons": results_sorted,
        "stats": {
            "expected_delta_depth": _summary_stats(exp_deltas),
            "ig_per_expected_delta_depth": _summary_stats(ratios),
        },
    }
    out_path.write_bytes(orjson.dumps(out_payload, option=orjson.OPT_INDENT_2))
    print(f"Wrote {out_path}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(args.report, payload=out_payload, top_n=int(args.top_n))
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
