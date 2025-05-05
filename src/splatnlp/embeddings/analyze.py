import logging
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import splatnlp.embeddings.analysis_helpers as helpers

logger = logging.getLogger(__name__)


def analyze_ability_importance(
    ability_lists: list[list[int]],
    cluster_labels: np.ndarray | None = None,
    inv_vocab: dict[int, str] | None = None,
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
) -> tuple[dict[int | str, dict[str, float]], dict[int | str, str]]:
    """
    Analyze and visualize the importance of abilities per cluster using TF-IDF
    scores.

    Orchestrates calls to helper functions for data preparation, scoring,
    collapsing, naming, and plotting.

    Args:
        ability_lists: List of ability sequences (lists of ability IDs).
        cluster_labels: Array of cluster labels. If None, assumes single
            cluster.
        inv_vocab: Optional mapping from ability IDs to names.
        top_n: Number of top abilities per cluster.
        plot: If True, generates bar charts.
        output_dir: Directory to save plots.
        filter_sub: If True, excludes abilities ending in '_3' or '_6'.
        name_clusters: If True, names clusters using top abilities.
        top_k: Limit analysis to top-k most frequent clusters.
        global_tfidf: Compute global TF-IDF instead of per-cluster.
        collapse_lower: Collapse similar abilities into one representative.
        collapse_threshold: Score threshold for naming/plotting inclusion.
        collapse_buffer: Buffer score for collapsing variants.

    Returns:
        Tuple: (final_cluster_scores, cluster_names)
            - final_cluster_scores: {cluster_id: {ability: score}}
            - cluster_names: {cluster_id: name_string}
    """
    cluster_labels, target_clusters, _ = helpers.prepare_cluster_data(
        ability_lists, cluster_labels, top_k
    )

    global_tfidf_data = None
    if global_tfidf:
        global_tfidf_data = helpers.compute_global_tfidf_data(
            ability_lists, inv_vocab
        )

    raw_cluster_scores: dict[int | str, dict[str, float]] = {}
    for cid in target_clusters:
        cluster_indices = np.where(cluster_labels == cid)[0]
        if len(cluster_indices) == 0:
            logger.debug(f"Skipping cluster {cid} as it has no members.")
            continue

        scores = helpers.calculate_cluster_scores(
            cluster_indices=cluster_indices,
            ability_lists=ability_lists,
            inv_vocab=inv_vocab,
            top_n=top_n,
            global_tfidf_data=global_tfidf_data,
        )

        top_scores = helpers.filter_and_select_top_abilities(
            scores, filter_sub, top_n
        )
        raw_cluster_scores[cid] = top_scores

    if collapse_lower:
        final_cluster_scores, cluster_names = helpers.collapse_best_variant(
            raw_cluster_scores,
            score_threshold=collapse_threshold,
            buffer=collapse_buffer,
        )
    else:
        final_cluster_scores = raw_cluster_scores
        cluster_names = helpers.generate_simple_cluster_names(
            final_cluster_scores, name_clusters, collapse_threshold
        )

    if plot:
        helpers.plot_cluster_abilities(
            final_cluster_scores, cluster_names, top_n, output_dir
        )

    return final_cluster_scores, cluster_names


def compare_clusters_by_abilities(
    cluster_importance: dict[Union[int, str], dict[str, float]],
    output_dir: Path | None = None,
    plot: bool = True,
    plot_figsize: tuple[float, float] | None = None,
    plot_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Compare clusters based on their important abilities via a heatmap.

    Creates a DataFrame and optional heatmap visualization showing the importance
    scores of abilities across different clusters. This helps identify which
    abilities are most characteristic of each cluster and how they compare
    between clusters.

    Args:
        cluster_importance: Dictionary mapping cluster IDs to dictionaries of
            ability scores. Format: {cluster_id: {ability: score}}
        output_dir: Optional directory to save the heatmap plot
        plot: Whether to generate and display the heatmap visualization
        plot_figsize: Optional tuple of (width, height) for the plot figure
        plot_kwargs: Optional dictionary of additional keyword arguments for
            seaborn's heatmap function

    Returns:
        DataFrame with abilities as rows, clusters as columns, and importance
        scores as values. Missing values are filled with 0.0.

    Example:
        >>> scores = {0: {"ability1": 0.8, "ability2": 0.5},
        ...           1: {"ability1": 0.3, "ability3": 0.7}}
        >>> df = compare_clusters_by_abilities(scores)
        >>> print(df)
                   0    1
        ability1  0.8  0.3
        ability2  0.5  0.0
        ability3  0.0  0.7
    """
    abs_set = sorted({a for sc in cluster_importance.values() for a in sc})
    cols = list(cluster_importance.keys())
    df = pd.DataFrame(index=abs_set, columns=cols).fillna(0.0)

    for cluster_id, scores_dict in cluster_importance.items():
        for ability, score in scores_dict.items():
            if ability in df.index:
                df.at[ability, cluster_id] = score

    if plot:
        figsize = plot_figsize or (12, max(8, len(df) * 0.3))
        plt.figure(figsize=figsize)

        hm_kw: dict[str, Any] = {
            "annot": True,
            "cmap": "YlGnBu",
            "linewidths": 0.5,
            "fmt": ".3f",
        }
        if plot_kwargs:
            hm_kw.update(plot_kwargs)

        sns.heatmap(df, **hm_kw)
        plt.title("Ability Importance Across Clusters")
        plt.xlabel("Cluster ID / Name")
        plt.ylabel("Ability")
        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "cluster_ability_comparison_heatmap.png"
            logger.info(f"Saving heatmap to: {filepath}")
            try:
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
            except Exception as e:
                logger.error(f"Failed to save heatmap {filepath}: {e}")

        plt.show()

    return df
