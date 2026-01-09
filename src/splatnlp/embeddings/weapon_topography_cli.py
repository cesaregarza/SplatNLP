"""Visualize Doc2Vec build topography for a single weapon.

This reads a tokenized dataset with columns:
  - ability_tags: JSON list of integer ability-token IDs
  - weapon_id: integer weapon tag (0..N-1)

It then:
  1) filters builds for a given weapon tag
  2) infers a Doc2Vec vector per build
  3) reduces to 2D via UMAP and clusters via DBSCAN
  4) saves static + interactive plots and a cluster summary JSON

Example (Snipewriter 5H):
    poetry run python -m splatnlp.embeddings.weapon_topography_cli \
      --weapon-token weapon_id_2070 \
      --model-path test_data/output_embeddings/doc2vec.model \
      --data-path test_data/tokenized/tokenized_data.csv \
      --vocab-path test_data/tokenized/vocab.json \
      --weapon-vocab-path test_data/tokenized/weapon_vocab.json \
      --out-dir tmp_results/doc2vec_topography
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd

from splatnlp.embeddings.clustering import cluster_vectors
from splatnlp.embeddings.inference import infer_doc2vec_vectors
from splatnlp.embeddings.load import load_doc2vec_model, load_vocab_json
from splatnlp.mechinterp.skill_helpers.context_loader import (
    _get_weapon_id_to_name,
)
from splatnlp.viz.cluster_labels import shorten


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return slug.strip("_") or "untitled"


def _reservoir_sample(
    items: list[str],
    sample: list[str],
    *,
    capacity: int,
    rng: random.Random,
    seen: int,
) -> int:
    """Update a reservoir sample in-place for a list of candidate items."""
    for item in items:
        seen += 1
        if len(sample) < capacity:
            sample.append(item)
            continue
        j = rng.randrange(seen)
        if j < capacity:
            sample[j] = item
    return seen


def load_weapon_builds(
    *,
    data_path: Path,
    weapon_tag: int,
    max_samples: int | None,
    seed: int,
    chunksize: int = 250_000,
) -> tuple[list[list[int]], int]:
    """Load (optionally sampled) ability tag lists for one weapon tag."""
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    rng = random.Random(int(seed))
    sampled_rows: list[str] = []
    seen = 0
    total_matches = 0

    reader = pd.read_csv(
        data_path,
        sep="\t",
        header=0,
        usecols=["ability_tags", "weapon_id"],
        dtype={"weapon_id": np.int16},
        chunksize=int(chunksize),
        low_memory=False,
    )
    for chunk in reader:
        matches = chunk.loc[chunk["weapon_id"] == weapon_tag, "ability_tags"]
        if matches.empty:
            continue
        rows = matches.tolist()
        total_matches += len(rows)

        if max_samples is None:
            sampled_rows.extend(rows)
        else:
            seen = _reservoir_sample(
                rows,
                sampled_rows,
                capacity=int(max_samples),
                rng=rng,
                seen=seen,
            )

    ability_lists: list[list[int]] = [orjson.loads(s) for s in sampled_rows]
    return ability_lists, int(total_matches)


def summarize_clusters(
    *,
    reduced_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    ability_lists: list[list[int]],
    inv_vocab: dict[int, str],
    cluster_names: dict[int, str] | None = None,
    cluster_tfidf_scores: dict[int, dict[str, float]] | None = None,
    top_tokens: int = 12,
    top_tfidf_tokens: int = 10,
) -> dict[str, Any]:
    if reduced_vectors.shape[0] != len(ability_lists):
        raise ValueError("reduced_vectors and ability_lists length mismatch")
    if cluster_labels.shape[0] != len(ability_lists):
        raise ValueError("cluster_labels and ability_lists length mismatch")

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_labels.tolist()):
        clusters[int(cid)].append(idx)

    total_builds = max(1, len(ability_lists))
    out_clusters: list[dict[str, Any]] = []
    for cid, indices in sorted(
        clusters.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    ):
        if cid == -1:
            cluster_name = "noise"
        else:
            cluster_name = (
                cluster_names.get(int(cid), str(cid))
                if cluster_names is not None
                else str(cid)
            )

        counter: Counter[int] = Counter()
        for idx in indices:
            counter.update(int(t) for t in ability_lists[idx])

        top = [
            {
                "token_id": int(tid),
                "token": inv_vocab.get(int(tid), str(tid)),
                "count": int(cnt),
            }
            for tid, cnt in counter.most_common(int(top_tokens))
        ]

        pts = reduced_vectors[np.array(indices, dtype=int)]
        center = pts.mean(axis=0)
        dist2 = ((pts - center) ** 2).sum(axis=1)
        rep_local = int(np.argmin(dist2))
        rep_idx = int(indices[rep_local])
        rep_tokens = [
            inv_vocab.get(int(t), str(t)) for t in ability_lists[rep_idx]
        ]

        tfidf_scores = (
            cluster_tfidf_scores.get(int(cid), {})
            if cluster_tfidf_scores is not None
            else {}
        )
        tfidf_top = sorted(
            tfidf_scores.items(), key=lambda kv: kv[1], reverse=True
        )[: int(top_tfidf_tokens)]

        out_clusters.append(
            {
                "cluster_id": int(cid),
                "cluster_name": cluster_name,
                "n": int(len(indices)),
                "pct": float(100.0 * len(indices) / total_builds),
                "center": [float(center[0]), float(center[1])],
                "representative_index": rep_idx,
                "representative_tokens": rep_tokens,
                "top_tokens": top,
                "top_tfidf_tokens": [
                    {"token": str(tok), "tfidf": float(score)}
                    for tok, score in tfidf_top
                ],
            }
        )

    return {
        "n_clusters": len([c for c in clusters if c != -1]),
        "n_noise": int(len(clusters.get(-1, []))),
        "cluster_sizes": {
            str(cid): int(len(ix)) for cid, ix in clusters.items()
        },
        "cluster_pcts": {
            str(cid): float(100.0 * len(ix) / total_builds)
            for cid, ix in clusters.items()
        },
        "clusters": out_clusters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weapon-token",
        type=str,
        required=True,
        help="Weapon token like weapon_id_2070 (Snipewriter 5H).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("test_data/output_embeddings/doc2vec.model"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("test_data/tokenized/tokenized_data.csv"),
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("test_data/tokenized/vocab.json"),
    )
    parser.add_argument(
        "--weapon-vocab-path",
        type=Path,
        default=Path("test_data/tokenized/weapon_vocab.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp_results/doc2vec_topography"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Max builds to sample for this weapon (0 = all).",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--dbscan-eps", type=float, default=0.5)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument(
        "--top-tokens",
        type=int,
        default=12,
        help="Top tokens per cluster in the JSON summary.",
    )
    parser.add_argument(
        "--annotate-top-k",
        type=int,
        default=25,
        help="Annotate the top-K clusters by size in the PNG (0 = all).",
    )
    parser.add_argument(
        "--png-point-size",
        type=float,
        default=60.0,
        help="Marker size (points^2) for the PNG scatter.",
    )
    parser.add_argument(
        "--png-point-alpha",
        type=float,
        default=0.75,
        help="Point alpha for the PNG scatter.",
    )
    parser.add_argument(
        "--plotly-point-size",
        type=int,
        default=6,
        help="Marker size (px) for the interactive HTML plot.",
    )
    args = parser.parse_args()

    weapon_vocab = load_vocab_json(args.weapon_vocab_path)
    if str(args.weapon_token) not in weapon_vocab:
        raise ValueError(
            f"Unknown weapon token {args.weapon_token} in "
            f"{args.weapon_vocab_path}"
        )
    weapon_tag = int(weapon_vocab[str(args.weapon_token)])

    numeric_id = str(args.weapon_token).split("_")[-1]
    label_map = _get_weapon_id_to_name()
    weapon_label = label_map.get(numeric_id, f"Weapon {numeric_id}")

    vocab = load_vocab_json(args.vocab_path)
    inv_vocab = {int(v): str(k) for k, v in vocab.items()}

    max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)
    ability_lists, total_matches = load_weapon_builds(
        data_path=args.data_path,
        weapon_tag=weapon_tag,
        max_samples=max_samples,
        seed=int(args.seed),
        chunksize=int(args.chunksize),
    )
    if not ability_lists:
        raise ValueError(
            f"No builds found for weapon_tag={weapon_tag} in {args.data_path}"
        )
    print(
        f"Loaded builds: n_used={len(ability_lists)} "
        f"(total_in_dataset={total_matches})"
    )

    model = load_doc2vec_model(args.model_path)
    vectors = infer_doc2vec_vectors(model, ability_lists)

    reduced, cluster_labels = cluster_vectors(
        vectors,
        eps=float(args.dbscan_eps),
        min_samples=int(args.dbscan_min_samples),
        umap_neighbors=int(args.umap_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        random_state=int(args.seed),
    )

    from splatnlp.embeddings.analyze import analyze_ability_importance

    tfidf_scores, tfidf_names = analyze_ability_importance(
        ability_lists=ability_lists,
        cluster_labels=cluster_labels,
        inv_vocab=inv_vocab,
        top_n=10,
        plot=False,
        name_clusters=True,
        collapse_lower=True,
        collapse_threshold=0.1,
        collapse_buffer=0.02,
        global_tfidf=True,
        filter_sub=True,
    )
    cluster_name_map = {
        int(cid): shorten(
            [p.strip() for p in str(name).split(",") if p.strip()],
            keep=3,
        )
        or str(cid)
        for cid, name in tfidf_names.items()
    }
    cluster_tfidf_map = {
        int(cid): {str(tok): float(score) for tok, score in scores.items()}
        for cid, scores in tfidf_scores.items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_slug(f"{weapon_label}_{args.weapon_token}_tag{weapon_tag}")
    np.save(args.out_dir / f"{base}_reduced.npy", reduced)
    np.save(args.out_dir / f"{base}_clusters.npy", cluster_labels)

    summary = summarize_clusters(
        reduced_vectors=reduced,
        cluster_labels=cluster_labels,
        ability_lists=ability_lists,
        inv_vocab=inv_vocab,
        cluster_names=cluster_name_map,
        cluster_tfidf_scores=cluster_tfidf_map,
        top_tokens=int(args.top_tokens),
    )

    summary_payload = {
        "meta": {
            "weapon_token": str(args.weapon_token),
            "weapon_tag": weapon_tag,
            "weapon_label": weapon_label,
            "data_path": str(args.data_path),
            "model_path": str(args.model_path),
            "total_matches_in_dataset": int(total_matches),
            "n_builds_used": int(len(ability_lists)),
            "seed": int(args.seed),
            "umap_neighbors": int(args.umap_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "tfidf_top_n": 10,
            "tfidf_collapse_threshold": 0.1,
            "tfidf_collapse_buffer": 0.02,
            "tfidf_filter_sub": True,
            "tfidf_global": True,
        },
        "summary": summary,
    }
    summary_path = args.out_dir / f"{base}_summary.json"
    summary_path.write_bytes(
        orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2)
    )

    # Plots (defer matplotlib/plotly imports so this module stays lightweight).
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import plotly.express as px
    from matplotlib import patheffects as pe

    from splatnlp.viz.setup_style import setup_style

    setup_style()

    # Static PNG.
    fig, ax = plt.subplots(figsize=(24, 16))
    labels = cluster_labels.astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    by_size = sorted(
        zip(unique.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True
    )
    top_for_legend = {cid for cid, _ in by_size[:10] if cid != -1}
    annotate_top_k = int(args.annotate_top_k)
    if annotate_top_k <= 0:
        annotate_clusters = {cid for cid, _ in by_size if cid != -1}
    else:
        annotate_clusters = {
            cid for cid, _ in by_size[:annotate_top_k] if cid != -1
        }

    noise_mask = labels == -1
    if noise_mask.any():
        noise_n = int(noise_mask.sum())
        noise_pct = 100.0 * noise_n / max(1, len(ability_lists))
        ax.scatter(
            reduced[noise_mask, 0],
            reduced[noise_mask, 1],
            s=max(1.0, float(args.png_point_size) * 0.5),
            c="lightgray",
            alpha=min(1.0, float(args.png_point_alpha) * 0.4),
            linewidths=0,
            label=f"noise (n={noise_n}, {noise_pct:.1f}%)",
        )

    cmap = plt.get_cmap("viridis")
    non_noise = [(int(cid), int(n)) for cid, n in by_size if int(cid) != -1]
    denom = max(1, len(non_noise) - 1)
    total_n = max(1, len(ability_lists))
    for rank, (cid, n) in enumerate(non_noise):
        cid = int(cid)
        mask = labels == cid
        color = cmap(float(rank) / float(denom))
        if cid in top_for_legend:
            name = cluster_name_map.get(cid, str(cid))
            pct = 100.0 * int(n) / total_n
            label_text = f"{cid}: {name} (n={int(n)}, {pct:.1f}%)"
        else:
            label_text = None
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=float(args.png_point_size),
            color=color,
            alpha=float(args.png_point_alpha),
            linewidths=0,
            label=label_text,
        )

        if cid in annotate_clusters:
            cx, cy = np.median(reduced[mask, :], axis=0).tolist()
            cname = cluster_name_map.get(cid, str(cid))
            txt = ax.text(
                float(cx),
                float(cy),
                str(cname),
                color="white",
                weight="bold",
                alpha=0.8,
                ha="center",
                va="center",
                fontsize=13,
                bbox=dict(
                    facecolor="black",
                    alpha=0.3,
                    edgecolor="black",
                ),
                zorder=6,
            )
            txt.set_path_effects(
                [pe.Stroke(linewidth=3, foreground="black"), pe.Normal()]
            )

    ax.set_title(
        f"{weapon_label} ({args.weapon_token}, tag={weapon_tag})\\n"
        f"builds={len(ability_lists)} (total={total_matches}), "
        f"clusters={summary['n_clusters']}, noise={summary['n_noise']}"
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    handles, labels_txt = ax.get_legend_handles_labels()
    handles_labels = [(h, t) for h, t in zip(handles, labels_txt) if t]
    if handles_labels:
        handles, labels_txt = zip(*handles_labels)
        ax.legend(
            handles,
            labels_txt,
            markerscale=2.5,
            fontsize=8,
            frameon=False,
            loc="best",
        )
    fig.tight_layout()
    png_path = args.out_dir / f"{base}_umap.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    # Interactive HTML (no per-point token hover to keep size reasonable).
    df_plot = pd.DataFrame(
        {
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "cluster_id": [int(c) for c in cluster_labels.tolist()],
        }
    )
    df_plot["cluster_name"] = df_plot["cluster_id"].map(
        lambda c: (
            "noise" if int(c) == -1 else cluster_name_map.get(int(c), str(c))
        )
    )
    fig_i = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster_id",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        hover_data=["cluster_id", "cluster_name"],
        title=(
            f"{weapon_label} ({args.weapon_token}) "
            "Doc2Vec build topography (UMAP)"
        ),
        opacity=0.6,
    )
    fig_i.update_traces(marker={"size": int(args.plotly_point_size)})
    html_path = args.out_dir / f"{base}_umap.html"
    fig_i.write_html(html_path)

    print(f"Wrote {png_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
