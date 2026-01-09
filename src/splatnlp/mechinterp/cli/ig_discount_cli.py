"""Postprocess an information-gain sweep JSON with a depth discount.

This reads an existing `info_gain_cli --all-weapons` JSON and applies a
depth-based discount to each weapon's information gain, without rerunning the
model.

Discount:
    ig_discount_factor = discount_base ** log(depth)
    information_gain_discounted = information_gain * ig_discount_factor

Where `depth` is the number of input tokens at the branch point
(`branch_context_len`).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Literal

import orjson

DepthLog = Literal["ln", "log2", "log10"]


def _depth_log(depth: int, *, mode: DepthLog) -> float:
    if depth <= 0:
        return 0.0
    if mode == "ln":
        return float(math.log(depth))
    if mode == "log2":
        return float(math.log2(depth))
    if mode == "log10":
        return float(math.log10(depth))
    raise ValueError(f"Unknown depth log mode: {mode}")


def _write_report(
    path: Path,
    *,
    payload: dict[str, Any],
    top_n: int,
) -> None:
    meta = payload.get("meta") or {}
    weapons = payload.get("weapons") or []

    def _fmt(x: float) -> str:
        return f"{float(x):.6f}"

    lines: list[str] = []
    lines.append("# Discounted Information Gain (Postprocess)")
    lines.append("")
    lines.append("Applies a depth discount to per-weapon information gain:")
    lines.append("")
    lines.append("- `ig_discount_factor = discount_base ** log(depth)`")
    lines.append(
        "- `information_gain_discounted = information_gain * ig_discount_factor`"
    )
    lines.append("")
    lines.append(
        "Where `depth = branch_context_len` at the greedy-closure branch point."
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Source: `{meta.get('source_path')}`")
    lines.append(f"- Discount base: `{meta.get('discount_base')}`")
    lines.append(f"- Depth log: `{meta.get('depth_log')}`")
    lines.append("")
    lines.append(f"## Top {top_n} Weapons By `information_gain_discounted`")
    lines.append("")
    lines.append(
        "| rank | weapon_id | weapon_label | IG | IG_discounted | "
        "discount | depth |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for i, row in enumerate(weapons[:top_n], start=1):
        lines.append(
            f"| {i} | `{row['weapon_id']}` | {row['weapon_label']} | "
            f"{_fmt(row['information_gain'])} | "
            f"{_fmt(row['information_gain_discounted'])} | "
            f"{_fmt(row['ig_discount_factor'])} | "
            f"{int(row['branch_context_len'])} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path(
            "tmp_results/"
            "info_gain_all_weapons_ultra_greedy0.7_"
            "omit_conditioned_next_tier_detailed.json"
        ),
        help="Path to an existing info_gain_cli --all-weapons JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to tmp_results/..._discounted.json).",
    )
    parser.add_argument("--discount-base", type=float, default=0.95)
    parser.add_argument(
        "--depth-log",
        type=str,
        default="ln",
        choices=["ln", "log2", "log10"],
        help="Which log to use for depth in the discount exponent.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional markdown report output path.",
    )
    args = parser.parse_args()

    payload = orjson.loads(args.in_path.read_bytes())
    meta = payload.get("meta") or {}
    weapons_in = payload.get("weapons") or []
    if not isinstance(weapons_in, list):
        raise ValueError("Unexpected payload shape (weapons is not a list)")

    discount_base = float(args.discount_base)
    if discount_base <= 0.0 or discount_base >= 1.0:
        raise ValueError("discount_base must be in (0, 1)")

    depth_log_mode: DepthLog = str(args.depth_log)  # type: ignore[assignment]

    weapons_out: list[dict[str, Any]] = []
    for row in weapons_in:
        wid = str(row.get("weapon_id"))
        label = str(row.get("weapon_label", ""))
        ig = float(row.get("information_gain", 0.0))

        depth = row.get("branch_context_len")
        if depth is None:
            ctx = row.get("branch_context") or []
            depth = len(ctx) if isinstance(ctx, list) else 1
        depth_i = max(1, int(depth))

        dlog = _depth_log(depth_i, mode=depth_log_mode)
        factor = float(discount_base**dlog)
        ig_disc = float(ig * factor)

        out_row = dict(row)
        out_row["branch_context_len"] = depth_i
        out_row["ig_discount_factor"] = factor
        out_row["information_gain_discounted"] = ig_disc
        weapons_out.append(out_row)

    weapons_out.sort(
        key=lambda r: float(r.get("information_gain_discounted", 0.0)),
        reverse=True,
    )

    top_n = max(0, int(args.top_n))
    print("Top weapons by information_gain_discounted:")
    for row in weapons_out[: min(10, len(weapons_out))]:
        print(
            f"{row['weapon_id']:14s} {str(row.get('weapon_label',''))[:28]:28s} "
            f"IGd={float(row['information_gain_discounted']):.6f} "
            f"IG={float(row['information_gain']):.6f} "
            f"disc={float(row['ig_discount_factor']):.4f} "
            f"depth={int(row['branch_context_len'])}"
        )

    out_path = (
        args.out
        if args.out is not None
        else args.in_path.with_name(
            args.in_path.name.replace(".json", "_discounted.json")
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_payload: dict[str, Any] = {
        "meta": {
            "source_path": str(args.in_path),
            "discount_base": discount_base,
            "depth_log": depth_log_mode,
            "source_meta": dict(meta) if isinstance(meta, dict) else {},
            "out_path": str(out_path),
        },
        "tokens": payload.get("tokens") or [],
        "weapons": weapons_out,
    }
    out_path.write_bytes(orjson.dumps(out_payload, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {out_path}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_report(args.report, payload=out_payload, top_n=top_n)
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
