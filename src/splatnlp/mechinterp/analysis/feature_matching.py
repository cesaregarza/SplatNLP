"""Cross-model feature matching utilities.

This module provides tools for matching features between the
Full (2K) and Ultra (24K) SAE models.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from splatnlp.mechinterp.skill_helpers.context_loader import (
    MechInterpContext,
    load_context,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatch:
    """A candidate match between features from different models."""

    source_feature: int
    source_model: str
    target_feature: int
    target_model: str

    # Match quality metrics
    activation_correlation: float = 0.0
    top_token_overlap: float = 0.0
    combined_score: float = 0.0

    # Evidence
    shared_top_tokens: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class MatchingReport:
    """Report from cross-model feature matching."""

    source_feature: int
    source_model: str
    target_model: str

    # Top matches
    matches: list[FeatureMatch] = field(default_factory=list)

    # Statistics
    n_candidates_tested: int = 0
    best_correlation: float = 0.0


class FeatureMatcher:
    """Matcher for finding corresponding features across models.

    Compares features from Full and Ultra models based on:
    - Activation correlation on shared examples
    - Top token overlap
    - PageRank similarity
    """

    def __init__(
        self,
        source_ctx: MechInterpContext,
        target_ctx: MechInterpContext | None = None,
    ):
        """Initialize matcher.

        Args:
            source_ctx: Context for source model
            target_ctx: Context for target model (if None, loads opposite model)
        """
        self.source_ctx = source_ctx

        if target_ctx is None:
            target_model = (
                "full" if source_ctx.model_type == "ultra" else "ultra"
            )
            self.target_ctx = load_context(target_model)
        else:
            self.target_ctx = target_ctx

    def find_matches(
        self,
        source_feature: int,
        n_candidates: int = 100,
        n_top_matches: int = 10,
    ) -> MatchingReport:
        """Find matching features in target model.

        Args:
            source_feature: Feature ID in source model
            n_candidates: Number of candidates to test
            n_top_matches: Number of top matches to return

        Returns:
            MatchingReport with candidate matches
        """
        logger.info(
            f"Finding matches for {self.source_ctx.model_type} "
            f"feature {source_feature} in {self.target_ctx.model_type}"
        )

        report = MatchingReport(
            source_feature=source_feature,
            source_model=self.source_ctx.model_type,
            target_model=self.target_ctx.model_type,
        )

        # Get source feature's top tokens
        source_top_tokens = self._get_top_tokens(
            self.source_ctx, source_feature
        )

        if not source_top_tokens:
            logger.warning(f"No data for source feature {source_feature}")
            return report

        # Get candidate features from target model
        target_features = self.target_ctx.db.get_all_feature_ids()
        if len(target_features) > n_candidates:
            # Sample candidates (could be smarter with prefiltering)
            import random

            random.seed(42)
            target_features = random.sample(target_features, n_candidates)

        # Score each candidate
        matches = []
        for target_fid in target_features:
            try:
                match = self._score_match(
                    source_feature, target_fid, source_top_tokens
                )
                if match.combined_score > 0:
                    matches.append(match)
            except Exception as e:
                logger.debug(f"Failed to score {target_fid}: {e}")

        # Sort by combined score
        matches.sort(key=lambda m: -m.combined_score)

        report.matches = matches[:n_top_matches]
        report.n_candidates_tested = len(target_features)
        if matches:
            report.best_correlation = matches[0].activation_correlation

        return report

    def _get_top_tokens(
        self,
        ctx: MechInterpContext,
        feature_id: int,
        top_n: int = 50,
    ) -> set[str]:
        """Get top tokens for a feature."""
        from collections import Counter

        try:
            data = ctx.db.get_feature_activations(feature_id, limit=1000)
        except Exception:
            return set()

        if not data:
            return set()

        # Sort by activation, take top 10%
        data = sorted(data, key=lambda x: x.get("activation", 0), reverse=True)
        n_top = max(1, len(data) // 10)
        top_data = data[:n_top]

        counter: Counter = Counter()
        for row in top_data:
            tokens = row.get("tokens", [])
            if tokens and isinstance(tokens[0], int):
                tokens = [ctx.inv_vocab.get(t, "") for t in tokens]

            for t in tokens:
                if t and not t.startswith("<"):
                    counter[t] += 1

        return set(tok for tok, _ in counter.most_common(top_n))

    def _score_match(
        self,
        source_fid: int,
        target_fid: int,
        source_top_tokens: set[str],
    ) -> FeatureMatch:
        """Score a potential feature match."""
        match = FeatureMatch(
            source_feature=source_fid,
            source_model=self.source_ctx.model_type,
            target_feature=target_fid,
            target_model=self.target_ctx.model_type,
        )

        # Get target's top tokens
        target_top_tokens = self._get_top_tokens(self.target_ctx, target_fid)

        if not target_top_tokens:
            return match

        # Compute token overlap (Jaccard)
        intersection = source_top_tokens & target_top_tokens
        union = source_top_tokens | target_top_tokens
        match.top_token_overlap = len(intersection) / len(union) if union else 0
        match.shared_top_tokens = list(intersection)[:10]

        # Compute activation correlation if token overlap is promising
        if match.top_token_overlap > 0.05:
            match.activation_correlation = self._compute_activation_correlation(
                source_fid, target_fid
            )

        # Combined score: weighted average of token overlap and activation corr
        # Token overlap is cheaper to compute so we use it for initial filtering
        # Activation correlation is more reliable for confirming matches
        if match.activation_correlation > 0:
            # Weight correlation more heavily when available
            match.combined_score = (
                0.4 * match.top_token_overlap
                + 0.6 * match.activation_correlation
            )
        else:
            match.combined_score = match.top_token_overlap

        # Add note about match quality
        if match.combined_score > 0.5:
            match.notes = "Strong match (high correlation + overlap)"
        elif match.combined_score > 0.3:
            match.notes = "Good match"
        elif match.combined_score > 0.1:
            match.notes = "Moderate match"
        else:
            match.notes = "Weak match"

        return match

    def _compute_activation_correlation(
        self,
        source_fid: int,
        target_fid: int,
        max_examples: int = 2000,
    ) -> float:
        """Compute Pearson correlation of activations on shared examples.

        Both models process the same input builds, so we can match examples
        by their input token sequences (ability builds).

        Args:
            source_fid: Feature ID in source model
            target_fid: Feature ID in target model
            max_examples: Maximum examples to use for correlation

        Returns:
            Pearson correlation coefficient (0.0 if insufficient data)
        """
        try:
            # Get activations from both models
            source_data = self.source_ctx.db.get_feature_activations(
                source_fid, limit=max_examples
            )
            target_data = self.target_ctx.db.get_feature_activations(
                target_fid, limit=max_examples
            )
        except Exception as e:
            logger.debug(f"Failed to load activations: {e}")
            return 0.0

        if not source_data or not target_data:
            return 0.0

        # Build lookup by token sequence (as hashable tuple)
        def make_build_key(row: dict) -> tuple | None:
            """Create hashable key from build tokens."""
            tokens = row.get("tokens", row.get("ability_input_tokens", []))
            if not tokens:
                return None
            # Sort for consistent ordering (builds are sets)
            return tuple(sorted(tokens))

        # Index source activations by build
        source_by_build: dict[tuple, float] = {}
        for row in source_data:
            key = make_build_key(row)
            if key:
                activation = row.get("activation", 0)
                # Use max activation if same build appears multiple times
                if key not in source_by_build:
                    source_by_build[key] = activation
                else:
                    source_by_build[key] = max(source_by_build[key], activation)

        # Index target activations by build
        target_by_build: dict[tuple, float] = {}
        for row in target_data:
            key = make_build_key(row)
            if key:
                activation = row.get("activation", 0)
                if key not in target_by_build:
                    target_by_build[key] = activation
                else:
                    target_by_build[key] = max(target_by_build[key], activation)

        # Find shared builds
        shared_keys = set(source_by_build.keys()) & set(target_by_build.keys())

        if len(shared_keys) < 30:
            logger.debug(
                f"Too few shared examples ({len(shared_keys)}) for correlation"
            )
            return 0.0

        # Extract paired activations
        source_acts = np.array([source_by_build[k] for k in shared_keys])
        target_acts = np.array([target_by_build[k] for k in shared_keys])

        # Compute Pearson correlation
        # Handle edge cases where variance is zero
        if np.std(source_acts) < 1e-10 or np.std(target_acts) < 1e-10:
            return 0.0

        correlation = np.corrcoef(source_acts, target_acts)[0, 1]

        # Handle NaN (can occur with degenerate data)
        if np.isnan(correlation):
            return 0.0

        # Return correlation (can be negative for anti-correlated features)
        return float(correlation)

    def compare_features(
        self,
        source_fid: int,
        target_fid: int,
    ) -> dict[str, Any]:
        """Detailed comparison of two specific features.

        Args:
            source_fid: Feature ID in source model
            target_fid: Feature ID in target model

        Returns:
            Dict with detailed comparison metrics
        """
        source_tokens = self._get_top_tokens(self.source_ctx, source_fid)
        target_tokens = self._get_top_tokens(self.target_ctx, target_fid)

        intersection = source_tokens & target_tokens
        source_only = source_tokens - target_tokens
        target_only = target_tokens - source_tokens

        jaccard = (
            len(intersection) / len(source_tokens | target_tokens)
            if source_tokens or target_tokens
            else 0
        )

        # Compute activation correlation for detailed comparison
        activation_corr = self._compute_activation_correlation(
            source_fid, target_fid
        )

        return {
            "source_feature": source_fid,
            "source_model": self.source_ctx.model_type,
            "target_feature": target_fid,
            "target_model": self.target_ctx.model_type,
            "shared_tokens": list(intersection),
            "source_only_tokens": list(source_only)[:20],
            "target_only_tokens": list(target_only)[:20],
            "jaccard_similarity": jaccard,
            "activation_correlation": activation_corr,
            "combined_score": (
                0.4 * jaccard + 0.6 * activation_corr
                if activation_corr > 0
                else jaccard
            ),
        }
