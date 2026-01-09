"""CLI for suppression/JS branchiness metrics (fixed next-tier support).

Example (all weapons, Ultra):
    poetry run python -m splatnlp.mechinterp.cli.branchiness_cli \
      --all-weapons \
      --stack-policy next-tier \
      --greedy-threshold 0.7 \
      --ig-json tmp_results/info_gain_all_weapons_ultra_greedy0.7_omit_conditioned_next_tier_detailed.json \
      --sendou-compare tmp_results/sendou_compare_train2-3_eval1_masks4-5-6_limit0_seed42_beam3_steps8_top3_condfix.json \
      --sendou-mask 6 \
      --out tmp_results/branchiness_all_weapons_ultra_greedy0.7_omit_conditioned_next_tier.json \
      --report docs/reports/branchiness_ultra_greedy0.7_omit_conditioned_next_tier_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch

from splatnlp.eval.sendou_baseline import ap_to_slot_items
from splatnlp.mechinterp.analysis.branchiness_metrics import (
    branchiness_for_weapon,
    fixed_support_distributions_for_weapon,
)
from splatnlp.mechinterp.skill_helpers.context_loader import (
    _get_weapon_id_to_name,
)
from splatnlp.model.models import SetCompletionModel


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


def _weapon_label(weapon_id: str, *, id_to_name: dict[str, str]) -> str:
    numeric_id = weapon_id.split("_")[-1]
    return id_to_name.get(numeric_id, f"Weapon {numeric_id}")


def _load_ig_frac(path: Path) -> dict[str, dict[str, float]]:
    payload = orjson.loads(path.read_bytes())
    weapons = payload.get("weapons") or []
    out: dict[str, dict[str, float]] = {}
    for row in weapons:
        wid = str(row.get("weapon_id"))
        h0 = float(row.get("h0", 0.0))
        ig = float(row.get("information_gain", 0.0))
        frac = float(ig / h0) if h0 > 0 else 0.0
        out[wid] = {"information_gain": ig, "ig_frac": frac}
    return out


def _multiset_is_subset(need: Counter[str], have: Counter[str]) -> bool:
    return not (need - have)


def _context_violation(
    observed: Counter[str], pred: Counter[str] | None
) -> float:
    if pred is None:
        return 1.0 if sum(observed.values()) > 0 else 0.0
    return 0.0 if _multiset_is_subset(observed, pred) else 1.0


def _sendou_context_violation_by_weapon(
    path: Path, *, method: str, mask: int
) -> dict[str, dict[str, float]]:
    payload = orjson.loads(path.read_bytes())
    cases = payload.get("cases") or []
    results = payload.get("results") or {}
    method_rows = (results.get(method) or {}).get("rows") or []

    case_by_id = {int(c["case_id"]): c for c in cases}
    viol: dict[str, list[float]] = defaultdict(list)
    for row in method_rows:
        if int(row.get("ability_mask", -1)) != int(mask):
            continue
        cid = int(row["case_id"])
        case = case_by_id.get(cid)
        if case is None:
            continue

        weapon_token = str(case["weapon_token"])
        observed = Counter(ap_to_slot_items(dict(case["masked_abilities_ap"])))
        pred_ap = row.get("predicted_top1_achieved_ap")
        pred = (
            None
            if pred_ap is None
            else Counter(ap_to_slot_items(dict(pred_ap)))
        )
        viol[weapon_token].append(_context_violation(observed, pred))

    out: dict[str, dict[str, float]] = {}
    for wid, vals in viol.items():
        out[wid] = {
            "n": float(len(vals)),
            "violation_rate": float(sum(vals) / len(vals)) if vals else 0.0,
        }
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("shape mismatch")
    if x.size == 0:
        return 0.0
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.sqrt((x0 * x0).sum() * (y0 * y0).sum()))
    return float((x0 * y0).sum() / denom) if denom > 0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_r(_rankdata(x), _rankdata(y))


def _slice_table(
    *,
    metric: np.ndarray,
    outcome: np.ndarray,
    weapon_ids: list[str],
    frac: float,
) -> dict[str, Any]:
    if metric.shape != outcome.shape:
        raise ValueError("shape mismatch")
    n = int(metric.size)
    if n == 0:
        return {}

    k = max(1, int(math.ceil(float(frac) * n)))
    order = np.argsort(metric)
    bottom_idx = order[:k]
    top_idx = order[-k:]

    def _summ(idx: np.ndarray) -> dict[str, Any]:
        return {
            "n_weapons": int(len(idx)),
            "metric_mean": float(metric[idx].mean()),
            "outcome_mean": float(outcome[idx].mean()),
            "weapons": [weapon_ids[int(i)] for i in idx.tolist()],
        }

    return {"bottom": _summ(bottom_idx), "top": _summ(top_idx)}


def _write_report(
    path: Path,
    *,
    payload: dict[str, Any],
) -> None:
    meta = payload.get("meta") or {}
    weapons = payload.get("weapons") or []
    eval_summary = payload.get("eval") or {}
    focus = payload.get("focus") or {}

    def _fmt(x: float) -> str:
        return f"{float(x):.6f}"

    lines: list[str] = []
    lines.append("# Branchiness Sweep (Ultra, Next-Tier Support)")
    lines.append("")
    lines.append(
        "Branchiness metrics computed at the first stalled point after"
    )
    lines.append("greedy closure, using a fixed candidate support.")
    lines.append("")
    lines.append("**Source JSON**")
    lines.append("")
    lines.append(f"- `{meta.get('out_path', '')}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Checkpoint: `{meta.get('checkpoint')}`")
    lines.append(f"- Device: `{meta.get('device')}`")
    lines.append(
        f"- Greedy closure threshold: `{meta.get('greedy_threshold')}`"
    )
    lines.append(f"- Omit conditioned tokens: `{meta.get('omit_conditioned')}`")
    lines.append(f"- Stack policy: `{meta.get('stack_policy')}`")
    lines.append(f"- Suppression top-M: `{meta.get('suppression_top_m')}`")
    lines.append(f"- Suppression weight: `{meta.get('suppression_weight')}`")
    lines.append(f"- JS log base: `{meta.get('log_base')}`")
    if meta.get("ig_json"):
        lines.append(f"- IG source: `{meta.get('ig_json')}`")
    lines.append("")

    by_suppress = sorted(
        weapons, key=lambda w: float(w["suppression_score"]), reverse=True
    )
    by_js = sorted(
        weapons, key=lambda w: float(w["branchiness_js"]), reverse=True
    )

    def _table(rows: list[dict[str, Any]], *, key: str, top_n: int) -> None:
        lines.append(f"## Top {top_n} Weapons By `{key}`")
        lines.append("")
        lines.append(
            "| rank | weapon_id | weapon_label | "
            f"{key} | H0 | branch_len | support0_len |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for i, row in enumerate(rows[:top_n], start=1):
            lines.append(
                f"| {i} | `{row['weapon_id']}` | {row['weapon_label']} | "
                f"{_fmt(row[key])} | {_fmt(row['h0'])} | "
                f"{int(row['branch_context_len'])} | "
                f"{int(row['support0_len'])} |"
            )
        lines.append("")

    _table(by_suppress, key="suppression_score", top_n=20)
    _table(by_js, key="branchiness_js", top_n=20)

    if eval_summary:
        lines.append("## Sendou Mask Evaluation (Context Violation)")
        lines.append("")
        lines.append(f"- Compare file: `{eval_summary.get('compare_path')}`")
        lines.append(f"- Method: `{eval_summary.get('method')}`")
        lines.append(f"- Mask: `{eval_summary.get('mask')}`")
        lines.append(f"- Min weapon n: `{eval_summary.get('min_n')}`")
        lines.append("")
        lines.append("| metric | pearson_r | spearman_rho | n_weapons |")
        lines.append("|---|---:|---:|---:|")
        for row in eval_summary.get("correlations", []):
            lines.append(
                f"| `{row['metric']}` | {_fmt(row['pearson_r'])} | "
                f"{_fmt(row['spearman_rho'])} | {int(row['n_weapons'])} |"
            )
        lines.append("")

    if focus:
        wid = str(focus.get("weapon_id", ""))
        lines.append(f"## Focus: `{wid}`")
        lines.append("")
        lines.append("### Branch context")
        lines.append("")
        lines.append(f"`{focus.get('branch_context', [])}`")
        lines.append("")
        lines.append("### Candidate probs (q)")
        lines.append("")
        lines.append("| token | q |")
        lines.append("|---|---:|")
        for tok, q in focus.get("p0_top", []):
            lines.append(f"| `{tok}` | {_fmt(q)} |")
        lines.append("")

        tokens = focus.get("tokens", [])
        ratio_matrix = focus.get("ratio_matrix", [])
        if tokens and ratio_matrix:
            lines.append("### Conditional coupling (q_t(u) / q(u))")
            lines.append("")
            lines.append(
                "| t \\ u | " + " | ".join(f"`{u}`" for u in tokens) + " |"
            )
            lines.append("|---|" + "---:|" * len(tokens))
            for row in ratio_matrix:
                t = row["t"]
                vals = row["ratios"]
                lines.append(
                    "| "
                    + f"`{t}` | "
                    + " | ".join(_fmt(v) for v in vals)
                    + " |"
                )
            lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--weapon-id", type=str)
    group.add_argument("--all-weapons", action="store_true")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--greedy-threshold", type=float, default=0.7)
    parser.add_argument(
        "--stack-policy",
        type=str,
        default="next-tier",
        choices=["none", "next-tier", "max-tier"],
    )
    parser.add_argument(
        "--include-conditioned",
        action="store_true",
        help="Include conditioned tokens in the candidate support.",
    )
    parser.add_argument("--suppression-top-m", type=int, default=8)
    parser.add_argument(
        "--suppression-weight",
        type=str,
        default="p0",
        choices=["p0", "uniform"],
    )
    parser.add_argument(
        "--js-top-k",
        type=int,
        default=15,
        help="How many per-token JS contributions to store.",
    )
    parser.add_argument("--log-base", type=float, default=2.0)
    parser.add_argument(
        "--ig-json",
        type=Path,
        default=None,
        help="Optional info_gain_all_weapons_*.json to attach IG_frac.",
    )
    parser.add_argument(
        "--sendou-compare",
        type=Path,
        default=None,
        help="Optional sendou_compare_*.json for correlation eval.",
    )
    parser.add_argument("--sendou-method", type=str, default="ultra")
    parser.add_argument("--sendou-mask", type=int, default=6)
    parser.add_argument("--sendou-min-n", type=int, default=10)
    parser.add_argument(
        "--focus-weapon",
        type=str,
        default="weapon_id_6011",
        help="Weapon id to include coupling matrix for.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional markdown report output path.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")

    vocab = _load_json(args.vocab)
    weapon_vocab = _load_json(args.weapon_vocab)
    model = _load_model(
        args.checkpoint, vocab=vocab, weapon_vocab=weapon_vocab, device=device
    )

    id_to_name = _get_weapon_id_to_name()
    ig_by_weapon = _load_ig_frac(args.ig_json) if args.ig_json else {}
    omit_conditioned = not bool(args.include_conditioned)

    if args.all_weapons:
        weapons_out: list[dict[str, Any]] = []
        for wid in sorted(weapon_vocab.keys()):
            label = _weapon_label(wid, id_to_name=id_to_name)
            res = branchiness_for_weapon(
                model=model,
                weapon_id=wid,
                vocab=vocab,
                weapon_vocab=weapon_vocab,
                batch_size=int(args.batch_size),
                log_base=float(args.log_base),
                greedy_threshold=float(args.greedy_threshold),
                omit_conditioned=omit_conditioned,
                stack_policy=str(args.stack_policy),
                suppression_top_m=int(args.suppression_top_m),
                suppression_weight=str(args.suppression_weight),
                js_top_k=int(args.js_top_k),
                device=device,
            )
            row: dict[str, Any] = {
                "weapon_id": wid,
                "weapon_label": label,
                "branch_context": list(res.branch_context),
                "branch_context_len": int(len(res.branch_context)),
                "support0_len": int(len(res.support0)),
                "h0": float(res.h0),
                "suppression_score": float(res.suppression_score),
                "branchiness_js": float(res.branchiness_js),
                "suppression_rows": [r.to_dict() for r in res.suppression_rows],
                "js_rows": [r.to_dict() for r in res.js_rows],
            }
            if wid in ig_by_weapon:
                row.update(ig_by_weapon[wid])
            weapons_out.append(row)

        weapons_out.sort(
            key=lambda w: float(w["suppression_score"]), reverse=True
        )
        print("Top weapons by suppression_score:")
        for row in weapons_out[: min(10, len(weapons_out))]:
            print(
                f"{row['weapon_id']:14s} {row['weapon_label'][:28]:28s} "
                f"SUP={float(row['suppression_score']):.6f} "
                f"JS={float(row['branchiness_js']):.6f} "
                f"H0={float(row['h0']):.4f}"
            )

        focus: dict[str, Any] | None = None
        focus_wid = str(args.focus_weapon) if args.focus_weapon else ""
        if focus_wid and focus_wid in weapon_vocab:
            dists = fixed_support_distributions_for_weapon(
                model=model,
                weapon_id=focus_wid,
                vocab=vocab,
                weapon_vocab=weapon_vocab,
                batch_size=int(args.batch_size),
                greedy_threshold=float(args.greedy_threshold),
                omit_conditioned=omit_conditioned,
                stack_policy=str(args.stack_policy),
                device=device,
            )
            tokens = [
                t
                for t in [
                    "special_charge_up_3",
                    "quick_respawn_3",
                    "special_saver_6",
                ]
                if t in dists.support0
            ]
            p0_sorted = sorted(
                dists.p0.items(), key=lambda kv: kv[1], reverse=True
            )[:15]

            ratio_matrix: list[dict[str, Any]] = []
            for t in tokens:
                ratios: list[float] = []
                for u in tokens:
                    if u == t:
                        ratios.append(1.0)
                        continue
                    base = float(dists.p0.get(u, 0.0))
                    cond = float(dists.conditional_probs.get(t, {}).get(u, 0.0))
                    ratios.append(float(cond / base) if base > 0 else 0.0)
                ratio_matrix.append({"t": t, "ratios": ratios})

            focus = {
                "weapon_id": focus_wid,
                "weapon_label": _weapon_label(focus_wid, id_to_name=id_to_name),
                "branch_context": list(dists.branch_context),
                "tokens": tokens,
                "p0_top": [(t, float(p)) for t, p in p0_sorted],
                "ratio_matrix": ratio_matrix,
            }

        eval_out: dict[str, Any] | None = None
        if args.sendou_compare is not None:
            by_weapon = _sendou_context_violation_by_weapon(
                args.sendou_compare,
                method=str(args.sendou_method),
                mask=int(args.sendou_mask),
            )
            min_n = int(args.sendou_min_n)

            joined: list[tuple[str, dict[str, Any], dict[str, float]]] = []
            for row in weapons_out:
                wid = str(row["weapon_id"])
                stats = by_weapon.get(wid)
                if stats is None:
                    continue
                n = int(stats.get("n", 0.0))
                if n < min_n:
                    continue
                joined.append((wid, row, stats))

            weapon_ids = [wid for wid, _, _ in joined]
            viol = np.array(
                [float(stats["violation_rate"]) for _, _, stats in joined],
                dtype=np.float64,
            )

            correlations: list[dict[str, Any]] = []
            for metric_key in [
                "suppression_score",
                "branchiness_js",
                "ig_frac",
                "information_gain",
            ]:
                if any(metric_key not in row for _, row, _ in joined):
                    continue
                vals = np.array(
                    [float(row[metric_key]) for _, row, _ in joined],
                    dtype=np.float64,
                )
                correlations.append(
                    {
                        "metric": metric_key,
                        "pearson_r": _pearson_r(vals, viol),
                        "spearman_rho": _spearman_rho(vals, viol),
                        "n_weapons": int(len(vals)),
                        "slice_10pct": _slice_table(
                            metric=vals,
                            outcome=viol,
                            weapon_ids=weapon_ids,
                            frac=0.1,
                        ),
                    }
                )

            eval_out = {
                "compare_path": str(args.sendou_compare),
                "method": str(args.sendou_method),
                "mask": int(args.sendou_mask),
                "min_n": int(min_n),
                "n_weapons": int(len(joined)),
                "correlations": correlations,
            }

        out_path = (
            args.out
            if args.out is not None
            else Path("tmp_results")
            / (
                "branchiness_all_weapons_ultra_"
                f"greedy{format(float(args.greedy_threshold), 'g')}_"
                f"{('include' if args.include_conditioned else 'omit')}_"
                "conditioned_"
                f"{args.stack_policy}.json"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "meta": {
                "checkpoint": str(args.checkpoint),
                "vocab": str(args.vocab),
                "weapon_vocab": str(args.weapon_vocab),
                "device": str(args.device),
                "base_context": ["<NULL>"],
                "greedy_threshold": float(args.greedy_threshold),
                "omit_conditioned": omit_conditioned,
                "stack_policy": str(args.stack_policy),
                "batch_size": int(args.batch_size),
                "log_base": float(args.log_base),
                "suppression_top_m": int(args.suppression_top_m),
                "suppression_weight": str(args.suppression_weight),
                "js_top_k": int(args.js_top_k),
                "ig_json": None if args.ig_json is None else str(args.ig_json),
                "out_path": str(out_path),
            },
            "weapons": weapons_out,
        }
        if focus is not None:
            payload["focus"] = focus
        if eval_out is not None:
            payload["eval"] = eval_out

        out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"\nWrote {out_path}")

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            _write_report(args.report, payload=payload)
            print(f"Wrote {args.report}")
        return

    assert args.weapon_id is not None
    wid = str(args.weapon_id)
    res = branchiness_for_weapon(
        model=model,
        weapon_id=wid,
        vocab=vocab,
        weapon_vocab=weapon_vocab,
        batch_size=int(args.batch_size),
        log_base=float(args.log_base),
        greedy_threshold=float(args.greedy_threshold),
        omit_conditioned=omit_conditioned,
        stack_policy=str(args.stack_policy),
        suppression_top_m=int(args.suppression_top_m),
        suppression_weight=str(args.suppression_weight),
        js_top_k=int(args.js_top_k),
        device=device,
    )
    print(
        f"weapon_id={res.weapon_id} SUP={res.suppression_score:.6f} "
        f"JS={res.branchiness_js:.6f} H0={res.h0:.6f}"
    )
    print(
        f"branch_context_len={len(res.branch_context)} "
        f"support0_len={len(res.support0)}"
    )

    if args.out is not None:
        payload = {
            "meta": {
                "weapon_id": wid,
                "checkpoint": str(args.checkpoint),
                "vocab": str(args.vocab),
                "weapon_vocab": str(args.weapon_vocab),
                "device": str(args.device),
                "base_context": ["<NULL>"],
                "greedy_threshold": float(args.greedy_threshold),
                "omit_conditioned": omit_conditioned,
                "stack_policy": str(args.stack_policy),
                "batch_size": int(args.batch_size),
                "log_base": float(args.log_base),
                "suppression_top_m": int(args.suppression_top_m),
                "suppression_weight": str(args.suppression_weight),
                "js_top_k": int(args.js_top_k),
            },
            "result": res.to_dict(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
