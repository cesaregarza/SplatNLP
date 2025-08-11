"""
Cached efficient database that adds influence data and histogram cache support to EfficientFSDatabase.
"""

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl

from splatnlp.dashboard.efficient_fs_database import EfficientFSDatabase

logger = logging.getLogger(__name__)


class CachedEfficientDatabase(EfficientFSDatabase):
    """
    Enhanced EfficientFSDatabase with support for precomputed influence data.
    """

    def __init__(
        self,
        data_dir: str = "/mnt/e/activations_ultra_efficient",
        examples_dir: str = "/mnt/e/dashboard_examples_optimized",
        num_bins: int = 20,
        influence_data_path: Optional[str] = None,
        precomputed_dir: Optional[str] = None,
    ):
        """
        Initialize cached efficient database.

        Args:
            data_dir: Path to Parquet/Zarr converted data
            examples_dir: Path to optimized examples storage
            num_bins: Number of bins for histograms
            influence_data_path: Optional path to precomputed influence data
            precomputed_dir: Optional directory with precomputed histograms/stats
        """
        super().__init__(data_dir, examples_dir, num_bins)

        self.influence_data = None
        self.precomputed_dir = (
            Path(precomputed_dir) if precomputed_dir else None
        )

        # Load influence data if available
        if influence_data_path:
            logger.info(f"Loading influence data from {influence_data_path}")
            try:
                if influence_data_path.endswith(".parquet"):
                    self.influence_data = pd.read_parquet(influence_data_path)
                else:
                    self.influence_data = pd.read_csv(influence_data_path)
                logger.info(
                    f"Loaded influence data for {len(self.influence_data)} features"
                )
            except Exception as e:
                logger.warning(f"Failed to load influence data: {e}")

        # Load precomputed histograms if available
        self._histograms_cache = {}
        if self.precomputed_dir and self.precomputed_dir.exists():
            histograms_path = self.precomputed_dir / "histograms.pkl"
            if histograms_path.exists():
                logger.info(
                    f"Loading precomputed histograms from {histograms_path}"
                )
                try:
                    with open(histograms_path, "rb") as f:
                        self._histograms_cache = pickle.load(f)
                    logger.info(
                        f"Loaded {len(self._histograms_cache)} precomputed histograms"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load precomputed histograms: {e}"
                    )

        # Load precomputed statistics if available
        self._stats_cache = {}
        if self.precomputed_dir and self.precomputed_dir.exists():
            stats_path = self.precomputed_dir / "feature_stats.pkl"
            if stats_path.exists():
                logger.info(f"Loading precomputed statistics from {stats_path}")
                try:
                    with open(stats_path, "rb") as f:
                        self._stats_cache = pickle.load(f)
                    logger.info(
                        f"Loaded {len(self._stats_cache)} precomputed statistics"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load precomputed statistics: {e}"
                    )

    def get_feature_influence(self, feature_id: int) -> Optional[dict]:
        """
        Get precomputed influence data for a feature.

        Returns dict with positive and negative influences or None if not available.
        """
        if self.influence_data is None:
            return None

        feature_data = self.influence_data[
            self.influence_data["feature_id"] == feature_id
        ]

        if feature_data.empty:
            return None

        row = feature_data.iloc[0]

        # Extract positive and negative influences
        positive = []
        negative = []

        for i in range(1, 31):  # Assuming top 30
            pos_tok_col = f"+{i}_tok"
            pos_val_col = f"+{i}_val"
            neg_tok_col = f"-{i}_tok"
            neg_val_col = f"-{i}_val"

            if (
                pos_tok_col in row
                and pd.notna(row[pos_tok_col])
                and row[pos_tok_col]
            ):
                positive.append(
                    {
                        "rank": i,
                        "token": row[pos_tok_col],
                        "value": row[pos_val_col],
                    }
                )

            if (
                neg_tok_col in row
                and pd.notna(row[neg_tok_col])
                and row[neg_tok_col]
            ):
                negative.append(
                    {
                        "rank": i,
                        "token": row[neg_tok_col],
                        "value": row[neg_val_col],
                    }
                )

        return {
            "feature_id": feature_id,
            "feature_label": row.get("feature_label", ""),
            "positive": positive,
            "negative": negative,
        }

    @lru_cache(maxsize=128)
    def get_feature_histogram(self, nid: int) -> pl.DataFrame:
        """
        Get histogram for a feature, using cache if available.
        """
        # Check if we have precomputed histogram
        if nid in self._histograms_cache:
            hist_data = self._histograms_cache[nid]
            hist_df = pl.DataFrame(
                {
                    "bin_idx": range(len(hist_data["counts"])),
                    "lower_bound": hist_data["lower_bounds"],
                    "upper_bound": hist_data["upper_bounds"],
                    "count": hist_data["counts"],
                }
            )
            return hist_df.sort("lower_bound", descending=True)

        # Fall back to parent's on-demand computation
        return super().get_feature_histogram(nid)

    def get_feature_stats(
        self, nid: int, with_zeros: bool = False
    ) -> dict[str, float]:
        """
        Get statistics for a feature, using cache if available.
        """
        # Check if we have precomputed stats
        if nid in self._stats_cache:
            return self._stats_cache[nid].copy()

        # Fall back to parent's computation
        return super().get_feature_stats(nid, with_zeros)
