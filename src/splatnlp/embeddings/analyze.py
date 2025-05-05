import logging
import re
from collections import Counter
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_suffix_re = re.compile(r"^(.*?)(?:_(\d+))?$")


def compute_tfidf_for_builds(
    ability_lists: list[list[int]],
    vocab: dict[str, int] = None,
    inv_vocab: dict[int, str] = None,
    min_df: int = 2,
    max_df: float = 0.95,
    top_n: int = 10,
) -> tuple[np.ndarray, list[str], list[dict[str, float]]]:
    """
    Compute TF-IDF scores for ability builds to identify distinctive abilities.
    Args:
        ability_lists: List of ability tag lists (each list contains integer ability IDs)
        vocab: Optional vocabulary mapping ability names to IDs
        inv_vocab: Optional inverse vocabulary mapping IDs to ability names
        min_df: Minimum document frequency for TF-IDF
        max_df: Maximum document frequency for TF-IDF (as a proportion)
        top_n: Number of top TF-IDF terms to return per build
    Returns:
        Tuple containing TF-IDF matrix, feature names, and per-build top-N scores dicts
    """
    logger.info(f"Computing per-cluster TF-IDF for {len(ability_lists)} builds")
    documents: list[str] = []
    for lst in ability_lists:
        if inv_vocab:
            documents.append(" ".join(inv_vocab.get(a, str(a)) for a in lst))
        else:
            documents.append(" ".join(str(a) for a in lst))

    vectorizer = TfidfVectorizer(
        min_df=min_df, max_df=max_df, analyzer="word", token_pattern=r"\S+"
    )
    try:
        X = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        logger.warning("No terms remain after TF-IDF pruning.")
        from scipy.sparse import csr_matrix

        return csr_matrix((len(documents), 0)), [], [{} for _ in documents]

    top_tfidf_by_build: list[dict[str, float]] = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[-top_n:][::-1]
        top_tfidf_by_build.append(
            {feature_names[j]: row[j] for j in top_idx if row[j] > 0}
        )

    return X, feature_names, top_tfidf_by_build


def analyze_ability_importance(
    ability_lists: list[list[int]],
    cluster_labels: np.ndarray | None = None,
    vocab: dict[str, int] = None,
    inv_vocab: dict[int, str] = None,
    top_n: int = 10,
    plot: bool = True,
    output_dir: Path | None = None,
    filter_sub: bool = False,
    name_clusters: bool = False,
    top_k: int | None = None,
    global_tfidf: bool = False,
    collapse_lower: bool = False,
    collapse_threshold: float = 0.1,
    collapse_buffer: float = 0.02,
) -> tuple[dict[Union[int, str], dict[str, float]], dict[Union[int, str], str]]:
    """
    Analyze ability importance per cluster, optionally using global IDF.
    Args:
        ability_lists: full list of builds
        cluster_labels: cluster assignment for each build
        vocab, inv_vocab: mappings
        top_n: top abilities per cluster
        plot: whether to plot bar charts
        output_dir: where to save plots
        filter_sub: remove abilities ending in _3 or _6
        name_clusters: use top ability as cluster key
        top_k: restrict to k most frequent clusters
        global_tfidf: if True, fit IDF on full corpus once
    Returns:
        tuple of:
        - Dictionary mapping cluster keys to top-N ability→score dicts
        - Dictionary mapping cluster keys to cluster names
    """
    if cluster_labels is None:
        cluster_labels = np.zeros(len(ability_lists), dtype=int)
    counts = Counter(cluster_labels)
    clusters = sorted(set(cluster_labels))
    if top_k:
        top_list = [c for c, _ in counts.most_common(top_k)]
        clusters = [c for c in clusters if c in top_list]
    # global TF-IDF setup
    if global_tfidf:
        docs_all = [
            " ".join(
                inv_vocab.get(a, str(a)) if inv_vocab else str(a) for a in lst
            )
            for lst in ability_lists
        ]
        gvec = TfidfVectorizer(min_df=2, max_df=0.95, token_pattern=r"\S+")
        X_full = gvec.fit_transform(docs_all)
        feats = gvec.get_feature_names_out()
    sub_pat = re.compile(r"_(3|6)$")
    scores_dict: dict[Union[int, str], dict[str, float]] = {}
    for cid in clusters:
        if cid == -1 and len(clusters) > 1:
            continue
        idx = np.where(cluster_labels == cid)[0]
        if global_tfidf:
            rows = X_full[idx, :]
            sums = np.asarray(rows.sum(axis=0)).ravel()
            avg = sums / len(idx)
            scores = {feats[i]: avg[i] for i in range(len(feats)) if avg[i] > 0}
        else:
            _, feats, tb = compute_tfidf_for_builds(
                [ability_lists[i] for i in idx],
                vocab=vocab,
                inv_vocab=inv_vocab,
                top_n=top_n,
            )
            agg = Counter()
            for d in tb:
                agg.update(d)
            scores = {k: v / len(idx) for k, v in agg.items()}
        if filter_sub:
            scores = {k: v for k, v in scores.items() if not sub_pat.search(k)}
        # select top_n
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        scores_dict[cid] = dict(top_items)
    # collapse lower if requested
    if collapse_lower:
        scores_dict, names = collapse_best_variant(
            scores_dict,
            score_threshold=collapse_threshold,
            buffer=collapse_buffer,
        )
    else:
        names = {}
        for cid, sc in scores_dict.items():
            if name_clusters:
                names[cid] = ",".join(
                    [a for a, v in sc.items() if v > collapse_threshold]
                ) or str(cid)
            else:
                names[cid] = str(cid)
    # plotting
    if plot:
        for cid, sc in scores_dict.items():
            if not sc:
                continue
            plt.figure(figsize=(10, 6))
            items = sorted(sc.items(), key=lambda x: x[1])
            ys, xs = zip(*items)
            sns.barplot(x=list(xs), y=list(ys), palette="viridis")
            plt.title(f"Top {top_n} for {names[cid]}")
            plt.tight_layout()
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    output_dir / f"cluster_{cid}_abilities.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()
    return scores_dict, names


def compare_clusters_by_abilities(
    cluster_importance: dict[Union[int, str], dict[str, float]],
    output_dir: Path | None = None,
    plot: bool = True,
    plot_figsize: tuple[float, float] | None = None,
    plot_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Compare clusters based on their important abilities.
    Args:
        cluster_importance: Dictionary mapping cluster IDs/names to ability->score
        output_dir: directory for saving heatmap
        plot: whether to plot
        plot_kwargs: extra seaborn.heatmap kwargs
    Returns:
        DataFrame of abilitiesxclusters heatmap
    """
    abs_set = sorted({a for sc in cluster_importance.values() for a in sc})
    cols = list(cluster_importance.keys())
    df = pd.DataFrame(index=abs_set, columns=cols).fillna(0)
    for c, sc in cluster_importance.items():
        for a, v in sc.items():
            df.at[a, c] = v
    # prefix zero
    df.index = [
        re.sub(r"_(3|6)$", lambda m: f"_0{m.group(1)}", a) for a in df.index
    ]
    df = df.sort_index()
    if plot:
        plt.figure(figsize=plot_figsize or (12, max(8, len(df) * 0.4)))
        hm_kw = {
            "annot": True,
            "cmap": "YlGnBu",
            "linewidths": 0.5,
            "fmt": ".3f",
        }
        if plot_kwargs:
            hm_kw.update(plot_kwargs)
        sns.heatmap(df, **hm_kw)
        plt.title("Ability Importance Across Clusters")
        plt.tight_layout()
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_dir / "cluster_ability_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
    return df


def collapse_best_variant(
    cluster_scores: dict[int | str, dict[str, float]],
    *,
    score_threshold: float = 0.1,
    buffer: float = 0.02,
) -> tuple[dict[int | str, dict[str, float]], dict[int | str, str]]:
    """
    Collapse numeric-suffix abilities so **one** variant per prefix survives.

    Rules
    -----
    • Choose the variant with the highest TF-IDF score.
    • If another variant's score is within `buffer` of that best score
      *and* has a larger numeric suffix, that variant wins instead.

    Parameters
    ----------
    cluster_scores : dict
        {cluster_id → {ability → TF-IDF score}}
    score_threshold : float, default 0.1
        Only abilities with score > threshold are included in the returned
        `cluster_names` string.
    buffer : float, default 0.02
        Tolerance for the “larger suffix can steal the spot” rule.

    Returns
    -------
    new_scores : dict
        Collapsed version of `cluster_scores`.
    cluster_names : dict
        {cluster_id → comma-separated string of surviving ability names
         (un-abbreviated, **no** length cap).}
    """
    new_scores: dict[int | str, dict[str, float]] = {}
    cluster_names: dict[int | str, str] = {}

    for cid, scores in cluster_scores.items():
        best_for: dict[str, tuple[str, float, int]] = {}

        for ability, sc in scores.items():
            base, num = _suffix_re.match(ability).groups()
            num_int = int(num) if num is not None else -1

            current = best_for.get(base)
            if current is None:
                best_for[base] = (ability, sc, num_int)
                continue

            _, best_sc, best_num = current

            if sc > best_sc + buffer:  # clearly better
                best_for[base] = (ability, sc, num_int)
            elif (
                best_sc - buffer <= sc <= best_sc + buffer
                and num_int > best_num
            ):

                best_for[base] = (ability, sc, num_int)

        collapsed = {ab: sc for ab, sc, _ in best_for.values()}
        new_scores[cid] = collapsed

        parts = [
            ab
            for ab, sc in sorted(
                collapsed.items(), key=lambda x: (-x[1], x[0])
            )
            if sc > score_threshold
        ]
        cluster_names[cid] = ", ".join(parts) or str(cid)

    return new_scores, cluster_names
