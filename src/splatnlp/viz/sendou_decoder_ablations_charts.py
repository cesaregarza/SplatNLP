from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import orjson
import pandas as pd

from splatnlp.viz.setup_style import COLORS, setup_style


def _load_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


LABEL_BY_METHOD: dict[str, str] = {
    "conditional": "Conditional mode",
    "random": "Random fill",
    "decoder_greedy_only": "Decoder greedy only",
    "decoder_beam1": "Decoder beam=1",
    "decoder_beam3": "Decoder beam=3",
    "decoder_beam1_strict_lock": "Decoder beam=1 (strict lock)",
    "decoder_beam3_strict_lock": "Decoder beam=3 (strict lock)",
    "oneshot_rank_until_full": "One-shot rank-until-full",
    "oneshot_threshold": "One-shot threshold",
    "oneshot_topk_fixed": "One-shot top-k fixed",
}

COLOR_BY_METHOD: dict[str, str] = {
    "conditional": COLORS.green,
    "random": COLORS.gray,
    "decoder_greedy_only": COLORS.orange,
    "decoder_beam1": COLORS.blue,
    "decoder_beam3": COLORS.purple,
    "decoder_beam1_strict_lock": COLORS.blue,
    "decoder_beam3_strict_lock": COLORS.purple,
    "oneshot_rank_until_full": COLORS.yellow,
    "oneshot_threshold": COLORS.red,
    "oneshot_topk_fixed": COLORS.pink,
}


def _summaries_df(stats_payload: dict[str, Any]) -> pd.DataFrame:
    reports = stats_payload.get("reports") or []
    if not isinstance(reports, list) or not reports:
        raise ValueError("Stats payload has no reports.")

    report = reports[0]
    all_section = report.get("all") or {}
    summaries = all_section.get("summaries") or []
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("Stats payload has no summaries.")

    df = pd.DataFrame(summaries)
    df["mask"] = df["mask"].astype(int)
    df["mean"] = df["mean"].astype(float)
    df["ci_low"] = df["ci_low"].astype(float)
    df["ci_high"] = df["ci_high"].astype(float)
    df["metric"] = df["metric"].astype(str)
    df["method"] = df["method"].astype(str)
    return df


def _plot_metric_vs_mask(
    df: pd.DataFrame,
    *,
    metric: str,
    methods: list[str],
    out_path: Path,
    title: str,
    ylabel: str,
    y_min: float | None = None,
    y_max: float | None = None,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    sub_all = df[df["metric"] == metric]
    masks = sorted(sub_all["mask"].unique().tolist())
    if sub_all.empty:
        raise ValueError(f"No rows found for metric={metric}")

    for method in methods:
        sub = sub_all[sub_all["method"] == method].sort_values("mask")
        if sub.empty:
            continue

        color = COLOR_BY_METHOD.get(method, COLORS.gray)
        label = LABEL_BY_METHOD.get(method, method)
        linestyle = "--" if method.endswith("_strict_lock") else "-"
        marker = "s" if method.endswith("_strict_lock") else "o"

        ax.plot(
            sub["mask"],
            sub["mean"],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            markersize=6,
            label=label,
            color=color,
        )
        ax.fill_between(
            sub["mask"],
            sub["ci_low"],
            sub["ci_high"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_title(title)
    ax.set_xlabel("Mask (number of hidden slots)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(masks)
    ax.set_xlim(min(masks), max(masks))
    ax.set_yscale(yscale)
    if y_min is None or y_max is None:
        y_lo = float(sub_all["ci_low"].min())
        y_hi = float(sub_all["ci_high"].max())
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
        if y_min is None:
            y_min = y_lo - pad
        if y_max is None:
            y_max = y_hi + pad
    ax.set_ylim(float(y_min), float(y_max))
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_wall_time_bar(
    compare_payload: dict[str, Any],
    *,
    methods: list[str],
    out_path: Path,
) -> None:
    results = compare_payload.get("results") or {}
    rows: list[dict[str, Any]] = []
    for method in methods:
        wall_time = results.get(method, {}).get("wall_time_s", None)
        if wall_time is None:
            continue
        rows.append({"method": method, "wall_time_s": float(wall_time)})

    if not rows:
        raise ValueError("No wall_time_s found in compare payload.")

    df = pd.DataFrame(rows).sort_values("wall_time_s", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLOR_BY_METHOD.get(m, COLORS.gray) for m in df["method"]]
    labels = [LABEL_BY_METHOD.get(m, m) for m in df["method"]]
    ax.barh(labels, df["wall_time_s"], color=colors)
    ax.set_xlabel("Wall time (seconds)")
    ax.set_title("Wall time by method (whole sweep)")
    ax.set_xscale("log")

    for y, val in enumerate(df["wall_time_s"].tolist()):
        ax.text(
            val,
            y,
            f" {val:.1f}s",
            va="center",
            ha="left",
            fontsize=10,
            color="white",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare",
        type=Path,
        default=Path(
            "tmp_results/sendou_decoder_ablations_"
            "mask1-2-3-4-5-6-7-8-9-10-11_seed42.json"
        ),
        help="Compare JSON produced by sendou_decoder_ablations.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(
            "tmp_results/sendou_decoder_ablations_"
            "mask1-2-3-4-5-6-7-8-9-10-11_seed42_stats.json"
        ),
        help="Stats JSON produced by sendou_stats.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "tmp_results/sendou_decoder_ablations_masks1-11_seed42_charts"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "decoder_beam3",
            "decoder_beam1",
            "oneshot_rank_until_full",
            "conditional",
            "random",
        ],
        help="Which methods to include in line charts.",
    )
    args = parser.parse_args()

    setup_style()

    compare_payload = _load_json(args.compare)
    stats_path = args.stats
    if not stats_path.exists():
        candidate = stats_path.with_name(stats_path.stem + "_v2.json")
        if candidate.exists():
            stats_path = candidate
    stats_payload = _load_json(stats_path)
    df = _summaries_df(stats_payload)
    available_metrics = set(df["metric"].unique().tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_vs_mask(
        df,
        metric="tier1_set_completion_slot_acc_top1",
        methods=list(args.methods),
        out_path=args.out_dir
        / "tier1_set_completion_slot_acc_top1_vs_mask.png",
        title="Tier-1 set completion slot acc (top-1) vs mask",
        ylabel="Accuracy",
        y_min=0.0,
        y_max=1.0,
    )
    _plot_metric_vs_mask(
        df,
        metric="top1_completion_slot_acc",
        methods=list(args.methods),
        out_path=args.out_dir / "top1_completion_slot_acc_vs_mask.png",
        title="Completion slot acc (top-1) vs mask",
        ylabel="Accuracy",
        y_min=0.0,
        y_max=1.0,
    )
    _plot_metric_vs_mask(
        df,
        metric="top1_context_violation",
        methods=list(args.methods),
        out_path=args.out_dir / "top1_context_violation_vs_mask.png",
        title="Context violation rate (top-1) vs mask",
        ylabel="Violation rate",
        y_min=0.0,
        y_max=1.0,
    )
    if "top1_cross_family_edit" in available_metrics:
        _plot_metric_vs_mask(
            df,
            metric="top1_cross_family_edit",
            methods=list(args.methods),
            out_path=args.out_dir / "top1_cross_family_edit_vs_mask.png",
            title="Cross-family lock violation rate (top-1) vs mask",
            ylabel="Violation rate",
            y_min=0.0,
            y_max=1.0,
        )
    if "top1_edit_cost" in available_metrics:
        _plot_metric_vs_mask(
            df,
            metric="top1_edit_cost",
            methods=list(args.methods),
            out_path=args.out_dir / "top1_edit_cost_vs_mask.png",
            title="Lock violation cost (top-1) vs mask",
            ylabel="Violation cost",
            y_min=0.0,
        )
    if "top1_nn_jaccard_family" in available_metrics:
        _plot_metric_vs_mask(
            df,
            metric="top1_nn_jaccard_family",
            methods=list(args.methods),
            out_path=args.out_dir / "top1_nn_jaccard_family_vs_mask.png",
            title="Nearest-neighbor Jaccard (family-space, top-1) vs mask",
            ylabel="Jaccard",
            y_min=0.0,
            y_max=1.0,
        )
    if "top1_pmi_frankenstein_penalty" in available_metrics:
        _plot_metric_vs_mask(
            df,
            metric="top1_pmi_frankenstein_penalty",
            methods=list(args.methods),
            out_path=args.out_dir / "top1_pmi_frankenstein_penalty_vs_mask.png",
            title="Frankenstein penalty (PMI, top-1) vs mask",
            ylabel="Penalty",
            y_min=0.0,
        )
    _plot_wall_time_bar(
        compare_payload,
        methods=list(LABEL_BY_METHOD.keys()),
        out_path=args.out_dir / "wall_time_s_by_method.png",
    )

    print(f"Wrote charts to {args.out_dir}")


if __name__ == "__main__":
    main()
