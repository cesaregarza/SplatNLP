import logging
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
ABILITY_SUFFIX_RE = re.compile(r"^(.*?)(?:_(\d+))?$")
SUB_PATTERN_RE = re.compile(r"_(3|6)$")


def compute_tfidf_for_abilities(
    builds: list[list[int]],
    inv_vocab: dict[int, str] | None = None,
    min_doc_freq: int = 2,
    max_doc_freq_ratio: float = 0.95,
    top_n_terms: int = 10,
) -> tuple[csr_matrix, list[str], list[dict[str, float]]]:
    """Compute TF-IDF scores for a list of ability builds."""
    logger.info(f"Computing TF-IDF for {len(builds)} builds")

    documents = [
        (
            " ".join(inv_vocab.get(ability, str(ability)) for ability in build)
            if inv_vocab
            else " ".join(str(ability) for ability in build)
        )
        for build in builds
    ]

    vectorizer = TfidfVectorizer(
        min_df=min_doc_freq,
        max_df=max_doc_freq_ratio,
        analyzer="word",
        token_pattern=r"\S+",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        ability_names = vectorizer.get_feature_names_out()
    except ValueError:
        logger.warning("No terms remain after TF-IDF filtering.")
        return csr_matrix((len(documents), 0)), [], [{} for _ in documents]

    top_terms_by_build = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i]
        if row.nnz > 0:
            row_data = row.data
            row_indices = row.indices
            sorted_indices = np.argsort(row_data)[::-1]
            top_n_actual = min(top_n_terms, len(sorted_indices))
            top_feature_indices = row_indices[sorted_indices[:top_n_actual]]
            top_scores = row_data[sorted_indices[:top_n_actual]]

            top_terms_by_build.append(
                {
                    ability_names[idx]: score
                    for idx, score in zip(top_feature_indices, top_scores)
                }
            )
        else:
            top_terms_by_build.append({})

    return tfidf_matrix, ability_names.tolist(), top_terms_by_build


def prepare_cluster_data(
    ability_lists: list[list[int]],
    cluster_labels: np.ndarray | None = None,
    top_k: int | None = None,
) -> tuple[np.ndarray, list[int | str], dict[int | str, int]]:
    """Prepares cluster labels, identifies target clusters, and counts members."""
    if cluster_labels is None:
        cluster_labels = np.zeros(len(ability_lists), dtype=int)

    cluster_counts = Counter(cluster_labels)
    valid_clusters = sorted(
        [c for c in cluster_counts if c != -1 or len(cluster_counts) == 1]
    )

    if top_k:
        top_clusters = [c for c, _ in cluster_counts.most_common() if c != -1]
        target_clusters = top_clusters[:top_k]
        if (
            -1 in cluster_counts
            and len(target_clusters) < top_k
            and len(cluster_counts) == 1
        ):
            target_clusters.append(-1)
        elif (
            -1 in cluster_counts
            and top_k
            and -1 in cluster_counts.most_common(top_k)
        ):
            if -1 not in target_clusters:
                target_clusters.append(-1)
        if not target_clusters and -1 in cluster_counts:
            target_clusters = [-1]
    else:
        target_clusters = valid_clusters

    target_cluster_counts = {
        cid: cluster_counts[cid] for cid in target_clusters
    }

    return cluster_labels, target_clusters, target_cluster_counts


def compute_global_tfidf_data(
    ability_lists: list[list[int]],
    inv_vocab: dict[int, str] | None = None,
) -> tuple[csr_matrix, list[str]]:
    """Computes TF-IDF across all builds."""
    logger.info("Computing global TF-IDF across all builds.")
    docs_all = [
        " ".join(inv_vocab.get(a, str(a)) if inv_vocab else str(a) for a in lst)
        for lst in ability_lists
    ]
    global_vectorizer = TfidfVectorizer(
        min_df=2, max_df=0.95, token_pattern=r"\S+"
    )
    try:
        global_tfidf_matrix = global_vectorizer.fit_transform(docs_all)
        global_features = global_vectorizer.get_feature_names_out().tolist()
        return global_tfidf_matrix, global_features
    except ValueError:
        logger.warning("No terms remain after global TF-IDF filtering.")
        return csr_matrix((len(docs_all), 0)), []


def calculate_cluster_scores(
    cluster_indices: np.ndarray,
    ability_lists: list[list[int]],
    inv_vocab: dict[int, str] | None,
    top_n: int,
    global_tfidf_data: tuple[csr_matrix, list[str]] | None,
) -> dict[str, float]:
    """Calculates the average TF-IDF scores for a single cluster."""
    num_builds_in_cluster = len(cluster_indices)
    if num_builds_in_cluster == 0:
        return {}

    if global_tfidf_data:
        global_tfidf_matrix, global_features = global_tfidf_data
        if global_tfidf_matrix.shape[1] == 0:
            return {}
        cluster_matrix_slice = global_tfidf_matrix[cluster_indices, :]
        avg_scores = np.asarray(cluster_matrix_slice.mean(axis=0)).ravel()
        scores = {
            global_features[i]: avg_scores[i]
            for i in range(len(global_features))
            if avg_scores[i] > 0
        }
    else:
        cluster_builds = [ability_lists[i] for i in cluster_indices]
        _, _, top_terms_per_build = compute_tfidf_for_abilities(
            builds=cluster_builds,
            inv_vocab=inv_vocab,
            top_n_terms=top_n * 2,
            min_doc_freq=1,
            max_doc_freq_ratio=1.0,
        )
        aggregated_scores = Counter()
        for term_dict in top_terms_per_build:
            aggregated_scores.update(term_dict)
        scores = {
            ability: total_score / num_builds_in_cluster
            for ability, total_score in aggregated_scores.items()
        }

    return scores


def filter_and_select_top_abilities(
    scores: dict[str, float],
    filter_sub: bool,
    top_n: int,
) -> dict[str, float]:
    """Filters scores (optional) and selects the top N."""
    if filter_sub:
        scores = {
            ability: score
            for ability, score in scores.items()
            if not SUB_PATTERN_RE.search(ability)
        }
    top_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
        :top_n
    ]
    return dict(top_items)


def collapse_best_variant(
    cluster_scores: dict[int | str, dict[str, float]],
    *,
    score_threshold: float = 0.1,
    buffer: float = 0.02,
) -> tuple[dict[int | str, dict[str, float]], dict[int | str, str]]:
    """
    Collapse numeric-suffix abilities so **one** variant per prefix survives.
    """
    new_scores: dict[int | str, dict[str, float]] = {}
    cluster_names: dict[int | str, str] = {}

    for cid, scores in cluster_scores.items():
        grouped_by_base: dict[str, list[tuple[str, float, int]]] = {}
        for ability, sc in scores.items():
            match = ABILITY_SUFFIX_RE.match(ability)
            if match:
                base, num_str = match.groups()
                num_int = int(num_str) if num_str is not None else -1
                if base not in grouped_by_base:
                    grouped_by_base[base] = []
                grouped_by_base[base].append((ability, sc, num_int))
            else:
                if ability not in grouped_by_base:
                    grouped_by_base[ability] = []
                grouped_by_base[ability].append((ability, sc, -1))

        best_for_base: dict[str, tuple[str, float, int]] = {}
        for base, variants in grouped_by_base.items():
            if not variants:
                continue
            variants.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best_ability, best_score, best_num = variants[0]
            for ability, score, num in variants[1:]:
                if score >= best_score - buffer and num > best_num:
                    best_ability, best_score, best_num = ability, score, num
            best_for_base[base] = (best_ability, best_score, best_num)

        collapsed_cluster_scores = {
            ability: score for ability, score, _ in best_for_base.values()
        }
        new_scores[cid] = collapsed_cluster_scores

        name_parts = [
            ability
            for ability, score in sorted(
                collapsed_cluster_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if score > score_threshold
        ]
        cluster_names[cid] = ", ".join(name_parts) or str(cid)

    return new_scores, cluster_names


def generate_simple_cluster_names(
    scores_dict: dict[int | str, dict[str, float]],
    name_clusters: bool,
    score_threshold: float,
) -> dict[int | str, str]:
    """Generates cluster names when collapsing is not performed."""
    if not name_clusters:
        return {cid: str(cid) for cid in scores_dict}

    cluster_names = {}
    for cid, scores in scores_dict.items():
        name_parts = [
            ability
            for ability, score in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )
            if score > score_threshold
        ]
        cluster_names[cid] = ", ".join(name_parts) or str(cid)
    return cluster_names


def plot_cluster_abilities(
    scores_dict: dict[int | str, dict[str, float]],
    cluster_names: dict[int | str, str],
    top_n: int,
    output_dir: Path | None = None,
) -> None:
    """Generates and optionally saves bar plots for top abilities per cluster."""
    logger.info(f"Generating plots for {len(scores_dict)} clusters.")
    for cid, scores in scores_dict.items():
        if not scores:
            logger.debug(
                f"Skipping plot for cluster {cid} as it has no scores."
            )
            continue

        plt.figure(figsize=(10, 6))
        items_to_plot = sorted(scores.items(), key=lambda item: item[1])
        abilities, plot_scores = zip(*items_to_plot)

        sns.barplot(
            x=list(plot_scores),
            y=list(abilities),
            palette="viridis",
            orient="h",
        )
        cluster_display_name = cluster_names.get(cid, str(cid))
        plt.title(
            f"Top {len(abilities)} Abilities for Cluster: "
            f"{cluster_display_name} (Max {top_n})"
        )
        plt.xlabel("Average TF-IDF Score")
        plt.ylabel("Abilities")
        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_filename = re.sub(
                r'[\\/*?:"<>|]', "", str(cluster_display_name)
            )
            safe_filename = safe_filename.replace(", ", "_").replace(" ", "_")
            max_len = 100
            safe_filename = (
                (safe_filename[:max_len] + "...")
                if len(safe_filename) > max_len
                else safe_filename
            )
            filename = f"cluster_{cid}_abilities_{safe_filename}.png"
            filepath = output_dir / filename
            logger.debug(f"Saving plot to: {filepath}")
            try:
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")

        plt.show()
