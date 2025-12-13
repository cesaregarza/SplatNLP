"""Find features similar to a given feature.

Uses token overlap and PageRank scores to identify related features.
"""

import logging
from collections import defaultdict
from typing import Literal

logger = logging.getLogger(__name__)


class SimilarFinder:
    """Find features similar to a given feature.

    Uses PageRank token distributions to compute similarity.
    """

    def __init__(self, model_type: Literal["full", "ultra"]):
        """Initialize the similar finder.

        Args:
            model_type: Model type for database access
        """
        self.model_type = model_type
        self._ctx = None
        self._db = None
        self._vocab = None
        self._inv_vocab = None
        self._token_distributions: dict[int, dict[str, float]] = {}

    @property
    def db(self):
        """Lazy load database."""
        if self._db is None:
            self._db = self.ctx.db
        return self._db

    @property
    def ctx(self):
        """Lazy load mechinterp context."""
        if self._ctx is None:
            from splatnlp.mechinterp.skill_helpers.context_loader import (
                load_context,
            )

            self._ctx = load_context(self.model_type)
        return self._ctx

    def _load_vocabs(self):
        """Load vocabularies."""
        if self._vocab is None:
            self._vocab = self.ctx.vocab
            self._inv_vocab = self.ctx.inv_vocab

    def _get_token_distribution(self, feature_id: int) -> dict[str, float]:
        """Get PageRank token distribution for a feature.

        Returns dict mapping token names to normalized scores.
        """
        if feature_id in self._token_distributions:
            return self._token_distributions[feature_id]

        self._load_vocabs()

        try:
            # Try to get precomputed top tokens from database
            summary = self.db.get_feature_summary(feature_id)
            if summary and "top_tokens" in summary:
                dist = {}
                total = 0
                for item in summary["top_tokens"][:20]:
                    token_id = item.get("token_id")
                    score = item.get("score", 0)
                    if token_id is not None:
                        token_name = self._inv_vocab.get(
                            token_id, f"t{token_id}"
                        )
                        dist[token_name] = score
                        total += score

                # Normalize
                if total > 0:
                    dist = {k: v / total for k, v in dist.items()}

                self._token_distributions[feature_id] = dist
                return dist
        except Exception as e:
            logger.debug(f"Could not get precomputed tokens: {e}")

        # Fall back to computing from activations
        try:
            from splatnlp.dashboard.utils.pagerank import PageRankAnalyzer

            df = self.db.get_feature_activations(feature_id, limit=5000)
            if df is None or len(df) == 0:
                return {}

            pr = PageRankAnalyzer(self._vocab, self._inv_vocab, mode="family")

            for row in df.iter_rows(named=True):
                token_ids = row.get("ability_input_tokens", [])
                activation = row.get("activation", 0)
                pr.add_example(token_ids, activation)

            scores = pr.compute_pagerank()
            top_tokens = pr.get_top_tokens(scores, top_k=20)

            dist = {}
            total = sum(score for _, _, score in top_tokens)
            if total > 0:
                for name, _, score in top_tokens:
                    dist[name] = score / total

            self._token_distributions[feature_id] = dist
            return dist

        except Exception as e:
            logger.warning(f"Error computing token distribution: {e}")
            return {}

    def _cosine_similarity(
        self, dist1: dict[str, float], dist2: dict[str, float]
    ) -> float:
        """Compute cosine similarity between two distributions."""
        if not dist1 or not dist2:
            return 0.0

        # Get all tokens
        all_tokens = set(dist1.keys()) | set(dist2.keys())

        # Compute dot product and magnitudes
        dot = 0.0
        mag1 = 0.0
        mag2 = 0.0

        for token in all_tokens:
            v1 = dist1.get(token, 0)
            v2 = dist2.get(token, 0)
            dot += v1 * v2
            mag1 += v1 * v1
            mag2 += v2 * v2

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1**0.5 * mag2**0.5)

    def find_by_top_tokens(
        self,
        feature_id: int,
        top_k: int = 10,
        candidate_features: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Find features with similar token distributions.

        Args:
            feature_id: Seed feature ID
            top_k: Number of similar features to return
            candidate_features: Optional list of feature IDs to search

        Returns:
            List of (feature_id, similarity) tuples, sorted by similarity
        """
        seed_dist = self._get_token_distribution(feature_id)
        if not seed_dist:
            logger.warning(
                f"Could not get distribution for feature {feature_id}"
            )
            return []

        # Get candidate features
        if candidate_features is None:
            try:
                candidate_features = self.db.get_all_feature_ids()
            except Exception as e:
                logger.warning(f"Could not load feature index: {e}")
                candidate_features = []

        # Compute similarities
        similarities = []
        for fid in candidate_features:
            if fid == feature_id:
                continue

            dist = self._get_token_distribution(fid)
            if dist:
                sim = self._cosine_similarity(seed_dist, dist)
                if sim > 0.1:  # Threshold for relevance
                    similarities.append((fid, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_by_family_overlap(
        self,
        feature_id: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find features with similar family distributions.

        Aggregates tokens by ability family before comparing.

        Args:
            feature_id: Seed feature ID
            top_k: Number of similar features to return

        Returns:
            List of (feature_id, similarity) tuples
        """
        from splatnlp.mechinterp.schemas.glossary import parse_token

        def aggregate_by_family(dist: dict[str, float]) -> dict[str, float]:
            """Aggregate token distribution by family."""
            family_scores = defaultdict(float)
            for token, score in dist.items():
                family, _ = parse_token(token)
                if family:
                    family_scores[family] += score
            return dict(family_scores)

        seed_dist = self._get_token_distribution(feature_id)
        if not seed_dist:
            return []

        seed_families = aggregate_by_family(seed_dist)

        # Get candidates
        try:
            candidates = self.db.get_all_feature_ids()
        except Exception:
            return []

        # Compute similarities
        similarities = []
        for fid in candidates:
            if fid == feature_id:
                continue

            dist = self._get_token_distribution(fid)
            if dist:
                families = aggregate_by_family(dist)
                sim = self._cosine_similarity(seed_families, families)
                if sim > 0.2:  # Family-level threshold
                    similarities.append((fid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_coactivating(
        self,
        feature_id: int,
        top_k: int = 10,
        min_coactivation: float = 0.1,
    ) -> list[tuple[int, float]]:
        """Find features that co-activate with the seed feature.

        Features that activate on similar examples.

        Args:
            feature_id: Seed feature ID
            top_k: Number of features to return
            min_coactivation: Minimum co-activation threshold

        Returns:
            List of (feature_id, coactivation_score) tuples
        """
        # This is a more expensive operation - would need to load
        # activation data for multiple features
        # For now, delegate to token-based similarity
        logger.info(
            "Co-activation analysis not yet implemented, "
            "using token similarity instead"
        )
        return self.find_by_top_tokens(feature_id, top_k=top_k)

    def get_similarity_report(
        self,
        feature_id: int,
        top_k: int = 5,
    ) -> dict:
        """Get a detailed similarity report for a feature.

        Args:
            feature_id: Feature to analyze
            top_k: Number of similar features per method

        Returns:
            Dict with similarity analysis
        """
        report = {
            "feature_id": feature_id,
            "model_type": self.model_type,
            "by_tokens": [],
            "by_family": [],
        }

        # Token-based similarity
        token_similar = self.find_by_top_tokens(feature_id, top_k=top_k)
        for fid, sim in token_similar:
            report["by_tokens"].append(
                {
                    "feature_id": fid,
                    "similarity": round(sim, 3),
                }
            )

        # Family-based similarity
        family_similar = self.find_by_family_overlap(feature_id, top_k=top_k)
        for fid, sim in family_similar:
            report["by_family"].append(
                {
                    "feature_id": fid,
                    "similarity": round(sim, 3),
                }
            )

        return report
