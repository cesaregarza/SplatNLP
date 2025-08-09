"""
Efficient filesystem-backed database using the optimized storage format.
Drop-in replacement for FSDatabase with the same API.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import orjson
import pandas as pd
import polars as pl
import zarr

from splatnlp.dashboard.utils.tfidf import compute_idf

logger = logging.getLogger(__name__)


class EfficientFSDatabase:
    """Efficient database using Parquet/Zarr storage."""

    def __init__(
        self,
        data_dir: str = "/mnt/e/activations_ultra_efficient",
        examples_dir: str = "/mnt/e/dashboard_examples_optimized",
        num_bins: int = 20,
    ):
        """
        Initialize the efficient database.

        Args:
            data_dir: Path to Parquet/Zarr converted data
            examples_dir: Path to optimized examples storage
            num_bins: Number of bins for histograms
        """
        self.data_dir = Path(data_dir)
        self.examples_dir = Path(examples_dir)
        self._nb_hist_bins = num_bins

        # Validate paths
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

        # Load metadata
        meta_path = self.data_dir / "conversion_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load optimized storage
        self.examples_df = None
        self.feature_index = None
        self.feature_lookup = {}

        # Load feature index
        index_path = self.examples_dir / "feature_index.parquet"
        if index_path.exists():
            self.feature_index = pl.read_parquet(index_path)
            logger.info(
                f"Loaded feature index with {len(self.feature_index)} features"
            )

            # Create lookup for fast access
            for row in self.feature_index.to_dicts():
                self.feature_lookup[row["feature_id"]] = (
                    row["start_idx"],
                    row["end_idx"],
                )

        # Load all examples (lazy)
        examples_path = self.examples_dir / "all_examples.parquet"
        if examples_path.exists():
            self.examples_df = pl.scan_parquet(examples_path)
            logger.info("Loaded examples storage (lazy)")

        # Cache for loaded batches
        self._batch_cache = {}

        # Load precomputed IDF if available
        idf_path = self.data_dir / "precomputed" / "idf.parquet"
        if idf_path.exists():
            self.idf = pl.read_parquet(idf_path)
            logger.info(f"Loaded precomputed IDF with {len(self.idf)} tokens")
        else:
            # Fallback to empty DataFrame if IDF not precomputed
            logger.warning(f"No precomputed IDF found at {idf_path}")
            logger.warning(
                "Run precompute_idf.py to generate IDF for better TF-IDF analysis"
            )
            self.idf = pl.DataFrame({"ability_input_tokens": [], "idf": []})

    # ---------- internal helpers -------------------------------------------

    def _get_batch_metadata(self, batch_idx: int) -> pl.DataFrame:
        """Load metadata for a batch (cached)."""
        if batch_idx not in self._batch_cache:
            parquet_path = (
                self.data_dir / "metadata" / f"batch_{batch_idx:04d}.parquet"
            )
            if parquet_path.exists():
                self._batch_cache[batch_idx] = pl.read_parquet(parquet_path)
            else:
                return pl.DataFrame()
        return self._batch_cache[batch_idx]

    @lru_cache(maxsize=32)
    def _load_feature_activations(
        self, feature_id: int
    ) -> tuple[np.ndarray, pl.DataFrame]:
        """
        Load all activations for a specific feature.

        Returns:
            Tuple of (activations array, metadata DataFrame)
        """
        all_acts = []
        all_meta = []

        for batch_idx in range(self.metadata.get("n_batches", 0)):
            # Load activations
            zarr_path = (
                self.data_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
            )
            if zarr_path.exists():
                z = zarr.open(str(zarr_path), mode="r")
                feature_acts = z[:, feature_id]

                # Get non-zero activations
                non_zero_mask = feature_acts > 0
                if non_zero_mask.any():
                    indices = np.where(non_zero_mask)[0]

                    # Get metadata for these samples
                    meta_df = self._get_batch_metadata(batch_idx)
                    if len(meta_df) > 0:
                        filtered_meta = meta_df.filter(
                            pl.col("sample_id").is_in(indices.tolist())
                        ).with_columns(
                            [
                                pl.lit(feature_acts[indices]).alias(
                                    "activation"
                                ),
                                pl.col("weapon_id_token").alias("weapon_id"),
                                pl.col("global_index").alias("index"),
                            ]
                        )

                        all_acts.append(feature_acts[non_zero_mask])
                        all_meta.append(filtered_meta)

        if all_acts:
            return np.concatenate(all_acts), pl.concat(all_meta)
        else:
            return np.array([]), pl.DataFrame()

    @lru_cache(maxsize=32)
    def _histogram_for(self, feature_id: int) -> pl.DataFrame:
        """Compute histogram for a feature."""
        acts, _ = self._load_feature_activations(feature_id)

        if len(acts) == 0:
            # Return empty histogram
            return pl.DataFrame(
                {
                    "bin_idx": [],
                    "lower_bound": [],
                    "upper_bound": [],
                    "count": [],
                }
            )

        counts, bins = np.histogram(acts, bins=self._nb_hist_bins)

        hist_df = pl.DataFrame(
            {
                "bin_idx": range(len(counts)),
                "lower_bound": bins[:-1],
                "upper_bound": bins[1:],
                "count": counts,
            }
        )

        return hist_df.sort("lower_bound", descending=True)

    @lru_cache(maxsize=32)
    def _stats_for(
        self, feature_id: int, with_zeros: bool = False
    ) -> dict[str, float]:
        """Compute statistics for a feature."""
        acts, _ = self._load_feature_activations(feature_id)

        total_samples = self.metadata.get("total_samples", 0)
        n_activations = len(acts)
        n_zeros = total_samples - n_activations

        if len(acts) == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "n_zeros": n_zeros,
                "n_total": total_samples,
                "sparsity": 1.0,
            }

        if with_zeros:
            # Include zeros in statistics
            acts = np.concatenate([acts, np.zeros(n_zeros)])

        return {
            "min": float(np.min(acts)),
            "max": float(np.max(acts)),
            "mean": float(np.mean(acts)),
            "median": float(np.median(acts)),
            "std": float(np.std(acts)),
            "q25": float(np.percentile(acts, 25)),
            "q75": float(np.percentile(acts, 75)),
            "n_zeros": n_zeros,
            "n_total": total_samples,
            "sparsity": (
                float(n_zeros / total_samples) if total_samples > 0 else 0.0
            ),
        }

    # ---------- public API expected by the components ----------------------

    def get_all_feature_ids(self) -> list[int]:
        """Get all feature IDs that have examples."""
        if self.feature_index is not None:
            return sorted(self.feature_index["feature_id"].to_list())
        else:
            # Default to all possible features
            return list(range(24576))

    def get_feature_histogram(self, feature_id: int) -> pl.DataFrame:
        """Get histogram for a feature."""
        return self._histogram_for(feature_id)

    def get_feature_activations(
        self, feature_id: int, limit: int | None = None
    ) -> pl.DataFrame:
        """Get top activations for a feature using optimized storage."""

        # Use optimized storage
        if feature_id in self.feature_lookup and self.examples_df is not None:
            start_idx, end_idx = self.feature_lookup[feature_id]

            # Use slice to get specific rows
            examples = self.examples_df.slice(start_idx, end_idx - start_idx)

            if limit:
                examples = examples.head(limit)

            # Collect and parse JSON
            df = examples.collect()

            # Parse ability tokens from JSON
            df = df.with_columns(
                [
                    pl.col("ability_tokens_json")
                    .map_elements(
                        lambda x: orjson.loads(x) if x else [],
                        return_dtype=pl.List(pl.Int64),
                    )
                    .alias("ability_tokens"),
                    pl.col("global_index").alias("index"),
                    pl.col("weapon_id").alias("weapon_id"),
                ]
            )

            # Rename columns to match expected format
            # Note: intervals grid expects 'ability_input_tokens' and 'weapon_id_token'
            # Top examples expects 'weapon_id', so we include both
            df = df.select(
                [
                    "index",
                    "activation",
                    pl.col("ability_tokens").alias("ability_input_tokens"),
                    pl.col("weapon_id").alias("weapon_id_token"),
                    pl.col("weapon_id").alias(
                        "weapon_id"
                    ),  # Keep original too for compatibility
                    "batch_id",
                    "sample_id",
                    "is_null",
                ]
            )

            return df

        # Fallback to loading from raw data if optimized storage not available
        _, meta_df = self._load_feature_activations(feature_id)

        if len(meta_df) > 0:
            df = meta_df.sort("activation", descending=True)
            if limit:
                df = df.head(limit)

            # Ensure column naming consistency
            if (
                "ability_tokens" in df.columns
                and "ability_input_tokens" not in df.columns
            ):
                df = df.rename({"ability_tokens": "ability_input_tokens"})

            # Also ensure weapon_id_token naming
            if (
                "weapon_id" in df.columns
                and "weapon_id_token" not in df.columns
            ):
                df = df.rename({"weapon_id": "weapon_id_token"})

            return df

        return pl.DataFrame()

    def get_feature_stats(
        self, feature_id: int, with_zeros: bool = False
    ) -> dict[str, float]:
        """Get statistics for a feature."""
        # Try to get stats from optimized index first
        if self.feature_index is not None:
            row = self.feature_index.filter(pl.col("feature_id") == feature_id)
            if len(row) > 0:
                stats_row = row.to_dicts()[0]

                # Calculate sparsity
                total_samples = self.metadata.get("total_samples", 842820)
                n_activations = stats_row.get("total_activations", 0)
                n_zeros = total_samples - n_activations

                stats = {
                    "min": 0.0,  # Min is always 0 for sparse features
                    "max": stats_row.get("max_activation", 0.0),
                    "mean": stats_row.get("mean_activation", 0.0),
                    "median": 0.0,  # Would need to calculate from raw data
                    "std": 0.0,  # Would need to calculate from raw data
                    "q25": 0.0,
                    "q75": 0.0,
                    "n_zeros": n_zeros,
                    "n_total": total_samples,
                    "sparsity": (
                        float(n_zeros / total_samples)
                        if total_samples > 0
                        else 1.0
                    ),
                }

                if with_zeros:
                    # Adjust mean to include zeros
                    stats["mean"] = stats["mean"] * (
                        n_activations / total_samples
                    )

                return stats

        # Fallback to computing from raw data
        return self._stats_for(feature_id, with_zeros)

    # ----- token / pair / triple examples ----------------------------------

    def get_single_token_examples(
        self, feature_id: int, limit: int = 50
    ) -> pd.DataFrame:
        """Get single token examples (converts from Polars to Pandas)."""
        df = self.get_feature_activations(feature_id, limit)

        if len(df) == 0:
            return pd.DataFrame()

        # Filter to single-token examples
        single_token_df = df.filter(
            pl.col("ability_input_tokens").list.len() == 1
        )

        return single_token_df.to_pandas()

    def get_top_examples(
        self, feature_id: int, limit: int = 50
    ) -> pd.DataFrame:
        """Get top examples (main method for dashboard)."""
        df = self.get_feature_activations(feature_id, limit)

        if len(df) == 0:
            return pd.DataFrame()

        return df.to_pandas()

    def get_triple_examples(
        self, feature_id: int, limit: int = 50
    ) -> pd.DataFrame:
        """Get triple token examples."""
        df = self.get_feature_activations(feature_id, limit)

        if len(df) == 0:
            return pd.DataFrame()

        # Filter to triple-token examples
        triple_df = df.filter(pl.col("ability_input_tokens").list.len() == 3)

        return triple_df.to_pandas()

    # ----- Additional methods for enhanced functionality -------------------

    def search_features_by_weapon(
        self, weapon_id: int, limit: int = 10
    ) -> List[int]:
        """Find features that activate for a specific weapon."""
        if self.feature_index is not None:
            # Use pre-computed index
            features = (
                self.feature_index.filter(
                    pl.col("top_weapons").list.contains(weapon_id)
                )
                .sort("max_activation", descending=True)
                .head(limit)
            )

            return features["feature_id"].to_list()

        return []

    def get_feature_summary(self, feature_id: int) -> Dict[str, Any]:
        """Get comprehensive summary for a feature."""
        stats = self.get_feature_stats(feature_id)

        # Add example count
        examples = self.get_feature_activations(feature_id, limit=100)
        stats["n_examples"] = len(examples)

        # Add weapon distribution
        if len(examples) > 0:
            weapon_counts = (
                examples.group_by("weapon_id")
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
                .head(5)
            )

            stats["top_weapons"] = weapon_counts.to_dicts()

        return stats
