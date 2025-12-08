"""Cluster analysis for multi-feature investigations.

This module provides utilities for analyzing relationships across
multiple SAE features, including co-activation patterns and
shared driver identification.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


@dataclass
class FeatureClusterReport:
    """Report from cluster analysis."""

    feature_ids: list[int]
    model_type: str

    # Co-activation matrix (feature_id -> feature_id -> correlation)
    coactivation_matrix: dict[int, dict[int, float]] = field(
        default_factory=dict
    )

    # Shared tokens across features
    shared_drivers: list[dict[str, Any]] = field(default_factory=list)

    # Feature groupings
    subclusters: list[list[int]] = field(default_factory=list)

    # Summary statistics
    mean_correlation: float = 0.0
    n_strong_pairs: int = 0


class ClusterAnalyzer:
    """Analyzer for multi-feature clusters.

    Investigates relationships between multiple SAE features to
    identify subsystems and shared structure.
    """

    def __init__(self, ctx: MechInterpContext):
        """Initialize with context."""
        self.ctx = ctx

    def analyze_cluster(
        self,
        feature_ids: list[int],
        sample_size: int = 5000,
    ) -> FeatureClusterReport:
        """Analyze a cluster of features.

        Args:
            feature_ids: List of feature IDs to analyze
            sample_size: Number of examples to sample

        Returns:
            FeatureClusterReport with analysis results
        """
        logger.info(f"Analyzing cluster of {len(feature_ids)} features")

        report = FeatureClusterReport(
            feature_ids=feature_ids,
            model_type=self.ctx.model_type,
        )

        # Load activation data for all features
        feature_activations = {}
        for fid in feature_ids:
            try:
                data = self.ctx.db.get_feature_activations(
                    fid, limit=sample_size
                )
                if data:
                    feature_activations[fid] = data
            except Exception as e:
                logger.warning(f"Failed to load feature {fid}: {e}")

        if len(feature_activations) < 2:
            logger.warning("Not enough features with data for cluster analysis")
            return report

        # Compute co-activation matrix
        report.coactivation_matrix = self._compute_coactivation(
            feature_activations, sample_size
        )

        # Find shared drivers
        report.shared_drivers = self._find_shared_drivers(feature_activations)

        # Identify subclusters
        report.subclusters = self._identify_subclusters(
            report.coactivation_matrix, threshold=0.5
        )

        # Summary stats
        correlations = []
        for fid1, corr_dict in report.coactivation_matrix.items():
            for fid2, corr in corr_dict.items():
                if fid1 < fid2:
                    correlations.append(corr)

        if correlations:
            report.mean_correlation = sum(correlations) / len(correlations)
            report.n_strong_pairs = sum(1 for c in correlations if c > 0.5)

        return report

    def _compute_coactivation(
        self,
        feature_activations: dict[int, list[dict]],
        sample_size: int,
    ) -> dict[int, dict[int, float]]:
        """Compute pairwise co-activation correlations."""
        from itertools import combinations

        # Build activation vectors by example index
        vectors: dict[int, dict[int, float]] = defaultdict(dict)

        for fid, data in feature_activations.items():
            for row in data:
                idx = row.get("index", row.get("idx", id(row)))
                act = row.get("activation", 0)
                vectors[idx][fid] = act

        # Compute correlations for each pair
        fids = list(feature_activations.keys())
        coact: dict[int, dict[int, float]] = defaultdict(dict)

        for fid1, fid2 in combinations(fids, 2):
            # Get paired values
            paired = [
                (v.get(fid1, 0), v.get(fid2, 0))
                for v in vectors.values()
                if fid1 in v and fid2 in v
            ]

            if len(paired) < 10:
                continue

            x = [p[0] for p in paired]
            y = [p[1] for p in paired]
            corr = self._pearson_correlation(x, y)

            coact[fid1][fid2] = round(corr, 3)
            coact[fid2][fid1] = round(corr, 3)

        return dict(coact)

    def _find_shared_drivers(
        self,
        feature_activations: dict[int, list[dict]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Find tokens that drive multiple features."""
        # Count tokens in high-activation examples for each feature
        feature_top_tokens: dict[int, Counter] = {}

        for fid, data in feature_activations.items():
            # Sort by activation, take top 10%
            sorted_data = sorted(
                data, key=lambda x: x.get("activation", 0), reverse=True
            )
            n_top = max(1, len(sorted_data) // 10)
            top_data = sorted_data[:n_top]

            counter: Counter = Counter()
            for row in top_data:
                tokens = row.get("tokens", [])
                if tokens and isinstance(tokens[0], int):
                    tokens = [self.ctx.inv_vocab.get(t, "") for t in tokens]

                for t in tokens:
                    if t and not t.startswith("<"):
                        counter[t] += 1

            feature_top_tokens[fid] = counter

        # Find tokens that appear in multiple features
        all_tokens = set()
        for counter in feature_top_tokens.values():
            all_tokens.update(counter.keys())

        shared = []
        for token in all_tokens:
            features_with_token = [
                fid
                for fid, counter in feature_top_tokens.items()
                if counter[token] > 0
            ]

            if len(features_with_token) > 1:
                total_count = sum(
                    feature_top_tokens[fid][token]
                    for fid in features_with_token
                )
                shared.append(
                    {
                        "token": token,
                        "n_features": len(features_with_token),
                        "feature_ids": features_with_token,
                        "total_count": total_count,
                    }
                )

        # Sort by number of features, then by count
        shared.sort(key=lambda x: (-x["n_features"], -x["total_count"]))
        return shared[:top_k]

    def _identify_subclusters(
        self,
        coact: dict[int, dict[int, float]],
        threshold: float = 0.5,
    ) -> list[list[int]]:
        """Identify subclusters using simple thresholding."""
        # Build adjacency based on threshold
        fids = list(coact.keys())
        adjacency: dict[int, set[int]] = defaultdict(set)

        for fid1, corr_dict in coact.items():
            for fid2, corr in corr_dict.items():
                if corr >= threshold:
                    adjacency[fid1].add(fid2)
                    adjacency[fid2].add(fid1)

        # Simple connected components
        visited = set()
        clusters = []

        for fid in fids:
            if fid in visited:
                continue

            cluster = []
            stack = [fid]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                stack.extend(adjacency[current] - visited)

            if cluster:
                clusters.append(sorted(cluster))

        return sorted(clusters, key=lambda x: -len(x))

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)
