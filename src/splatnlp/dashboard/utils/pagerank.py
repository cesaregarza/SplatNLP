"""PageRank analysis for token co-occurrence graphs."""

import re
from collections import defaultdict
from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy.sparse import lil_matrix

from splatnlp.utils.constants import MAIN_ONLY_ABILITIES

# Pattern for standard ability tokens: ability_family_AP (e.g., swim_speed_up_21)
ABILITY_FAMILY_RE = re.compile(r"^([a-z_]+?)_(\d+)$")
# Pattern to find ability suffix in compound tokens (ends with _XX where XX is digits)
ABILITY_SUFFIX_RE = re.compile(r"^(.+)_([a-z_]+?_\d+)$")
MAIN_ONLY_AP = 10


class PageRankAnalyzer:
    """Builds co-occurrence graphs and computes PageRank scores."""

    def __init__(
        self,
        vocab: Dict[str, int],
        inv_vocab: Dict[int, str],
        mode: Literal["raw", "ap_weighted", "family"] = "raw",
    ):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.mode = mode
        self.vocab_size = len(vocab)

        # For family mode, build family mappings
        if mode == "family":
            self._build_family_mappings()
            matrix_size = len(self._family_to_idx)
        else:
            matrix_size = self.vocab_size

        # lil_matrix for efficient incremental construction
        self._adj_matrix = lil_matrix(
            (matrix_size, matrix_size), dtype=np.float64
        )
        self._ap_cache = self._build_ap_cache()

    def _build_family_mappings(self) -> None:
        """Build mappings from tokens to family names."""
        self._token_to_family = {}  # token_id -> family_name
        self._family_to_idx = {}  # family_name -> family_idx
        self._idx_to_family = {}  # family_idx -> family_name

        families_seen = set()

        for token_id, token_str in self.inv_vocab.items():
            family = self._get_family_name(token_str)
            if family:
                self._token_to_family[token_id] = family
                families_seen.add(family)

        # Assign indices to families
        for idx, family in enumerate(sorted(families_seen)):
            self._family_to_idx[family] = idx
            self._idx_to_family[idx] = family

    def _get_family_name(self, token: str) -> str | None:
        """Extract family name from a token.

        Handles both regular tokens (swim_speed_up_21) and compound tokens
        (Splat_Bomb_swim_speed_up_21, Trizooka_special_charge_up_3, etc).
        """
        if token.startswith("<"):
            return None

        # First try: regular ability token (swim_speed_up_21)
        match = ABILITY_FAMILY_RE.match(token)
        if match:
            return match.group(1)  # e.g., "swim_speed_up"

        # Check if it's a main-only ability
        if token in MAIN_ONLY_ABILITIES:
            return token

        # Try compound token: Category_ability_XX format
        # Find ability suffix by looking for pattern ending in _digits
        compound_match = ABILITY_SUFFIX_RE.match(token)
        if compound_match:
            category_prefix = compound_match.group(1)  # e.g., "Splat_Bomb"
            ability_part = compound_match.group(2)  # e.g., "swim_speed_up_21"

            # Extract family from ability part
            ability_match = ABILITY_FAMILY_RE.match(ability_part)
            if ability_match:
                # Return compound family: Category_ability_family
                return f"{category_prefix}_{ability_match.group(1)}"

        # Check for main-only compound: Category_ninja_squid
        for main_only in MAIN_ONLY_ABILITIES:
            if token.endswith(f"_{main_only}"):
                return token  # The whole token is the family

        return None

    def _build_ap_cache(self) -> Dict[int, int]:
        """Pre-compute AP values for all tokens."""
        cache = {}
        for token_id, token_str in self.inv_vocab.items():
            cache[token_id] = self._get_ap_value(token_str)
        return cache

    def _get_ap_value(self, token: str) -> int:
        """Get AP value for a token string.

        Handles both regular tokens and compound tokens.
        """
        if token.startswith("<"):
            return 0

        # First try: regular ability token (swim_speed_up_21)
        match = ABILITY_FAMILY_RE.match(token)
        if match:
            return int(match.group(2))

        # Check if it's a main-only ability
        if token in MAIN_ONLY_ABILITIES:
            return MAIN_ONLY_AP

        # Try compound token: Category_ability_XX format
        compound_match = ABILITY_SUFFIX_RE.match(token)
        if compound_match:
            ability_part = compound_match.group(2)  # e.g., "swim_speed_up_21"
            ability_match = ABILITY_FAMILY_RE.match(ability_part)
            if ability_match:
                return int(ability_match.group(2))

        # Check for main-only compound: Category_ninja_squid
        for main_only in MAIN_ONLY_ABILITIES:
            if token.endswith(f"_{main_only}"):
                return MAIN_ONLY_AP

        return 0

    def add_example(self, token_ids: List[int], activation: float) -> None:
        """Add edges for all token pairs in an example."""
        if len(token_ids) < 2:
            return

        if self.mode == "family":
            self._add_example_family(token_ids, activation)
        else:
            self._add_example_tokens(token_ids, activation)

    def _add_example_tokens(self, token_ids: List[int], activation: float) -> None:
        """Add edges for token-level modes (raw or ap_weighted)."""
        for i, u in enumerate(token_ids):
            if u >= self.vocab_size:
                continue
            for j, v in enumerate(token_ids):
                if i == j or v >= self.vocab_size:
                    continue

                if self.mode == "raw":
                    weight = 1.0 * activation
                else:  # ap_weighted mode
                    weight = self._ap_cache.get(v, 0) * activation

                self._adj_matrix[u, v] += weight

    def _add_example_family(self, token_ids: List[int], activation: float) -> None:
        """Add edges for family-collapsed mode.

        For each family, only the highest AP token is used.
        E.g., if swim_speed_up_3 and swim_speed_up_6 co-occur, only use 6.
        """
        # Map families to their max AP token in this example
        family_max_ap: Dict[str, int] = {}  # family -> max AP value

        for token_id in token_ids:
            family = self._token_to_family.get(token_id)
            if not family:
                continue

            ap = self._ap_cache.get(token_id, 0)
            if family not in family_max_ap or ap > family_max_ap[family]:
                family_max_ap[family] = ap

        # Add edges between all family pairs (using max AP as weight)
        family_list = list(family_max_ap.keys())
        for i, fam_u in enumerate(family_list):
            idx_u = self._family_to_idx[fam_u]
            for j, fam_v in enumerate(family_list):
                if i == j:
                    continue
                idx_v = self._family_to_idx[fam_v]
                # Weight by receiving family's max AP
                weight = family_max_ap[fam_v] * activation
                self._adj_matrix[idx_u, idx_v] += weight

    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Compute PageRank scores using power iteration."""
        adj = self._adj_matrix.tocsr()
        n = adj.shape[0]

        # Handle empty matrix case
        if n == 0:
            return np.array([])

        # Row-normalize for transition probabilities
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        transition = adj.multiply(1.0 / row_sums[:, np.newaxis])

        # Power iteration
        pr = np.ones(n) / n
        for _ in range(max_iter):
            new_pr = (1 - damping) / n + damping * transition.T.dot(pr)
            if np.abs(new_pr - pr).sum() < tol:
                break
            pr = new_pr

        return pr

    def get_top_tokens(
        self, scores: np.ndarray, top_k: int = 20
    ) -> List[Tuple[str, int, float]]:
        """Get top tokens/families sorted by PageRank score."""
        sorted_indices = np.argsort(scores)[::-1]
        results = []

        if self.mode == "family":
            # Return family names
            for idx in sorted_indices:
                if len(results) >= top_k:
                    break
                family = self._idx_to_family.get(idx)
                if family:
                    results.append((family, idx, float(scores[idx])))
        else:
            # Return individual tokens
            for idx in sorted_indices:
                if len(results) >= top_k:
                    break
                token = self.inv_vocab.get(idx, f"<UNK_{idx}>")
                if not token.startswith("<"):
                    results.append((token, idx, float(scores[idx])))

        return results

    def reset(self) -> None:
        """Clear adjacency matrix for new analysis."""
        if self.mode == "family":
            matrix_size = len(self._family_to_idx)
        else:
            matrix_size = self.vocab_size

        self._adj_matrix = lil_matrix(
            (matrix_size, matrix_size), dtype=np.float64
        )
