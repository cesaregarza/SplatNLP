"""
Filesystem-backed data access for the SAE dashboard.

The public methods deliberately mirror those of DuckDBDatabase so that none
of the UI callbacks have to change.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from splatnlp.dashboard.utils.tfidf import compute_idf

logger = logging.getLogger(__name__)


class FSDatabase:
    # ---------- construction ------------------------------------------------
    def __init__(
        self,
        meta_path: str = "/mnt/e/activations2/outputs/",
        neurons_root: str = "/mnt/e/activations2/outputs/neuron_acts",
        num_bins: int = 20,
    ):
        """Initialize the filesystem-based database.

        Args:
            meta_path: Path to the metadata joblib file
            neurons_root: Path to the root directory containing neuron_XXXX folders
        """
        self.meta_path = Path(meta_path)
        self.neurons_root = Path(neurons_root)

        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)
        if not self.neurons_root.exists():
            raise FileNotFoundError(self.neurons_root)

        self.analysis_df = (
            pl.read_ipc(self.meta_path / "analysis_df_records.ipc")
            .drop(["is_null_token", "weapon_name"])
            .with_row_index("index")
        )
        self.metadata = json.load(open(self.meta_path / "metadata.json"))
        self.idf = compute_idf(self.analysis_df)

        # Find all neuron directories
        self.neuron_dirs = {
            int(d.name.split("_")[1]): d
            for d in self.neurons_root.iterdir()
            if d.is_dir() and d.name.startswith("neuron_")
        }
        logger.info(f"Found {len(self.neuron_dirs)} neuron directories")

        # Number of bins for histograms
        self._nb_hist_bins = num_bins

    # ---------- internal helpers -------------------------------------------
    @lru_cache(maxsize=None)
    def _load_single_neuron_df(self, nid: int) -> pl.DataFrame:
        """Materialise a long-form dataframe for one neuron (cached)."""

        n_dir = self.neuron_dirs[nid]
        idxs: np.ndarray = np.load(n_dir / "idxs.npy")
        acts: np.ndarray = np.load(n_dir / "acts.npy")
        acts_df = pl.DataFrame(
            {"index": idxs, "activation": acts.astype(np.float32)}
        )

        return (
            self.analysis_df.join(acts_df, on="index", how="inner")
            .with_columns(pl.col("weapon_id_token").alias("weapon_id"))
            .sort(["index", "weapon_id"])
        )

    @lru_cache(maxsize=None)
    def _histogram_for(self, nid: int) -> pl.DataFrame:
        acts = self._load_single_neuron_df(nid)["activation"].to_numpy()
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

    @lru_cache(maxsize=None)
    def _stats_for(
        self, nid: int, with_zeros: bool = False
    ) -> dict[str, float]:
        s = self._load_single_neuron_df(nid)["activation"]
        # s already has zeros removed, so we use metadata to get the total count
        total_count = self.metadata["num_regular"] + self.metadata["num_null"]
        num_zeros = total_count - len(s)
        if with_zeros:
            # Use polars concat instead of numpy
            s = pl.concat([s, pl.Series("activation", [0.0] * num_zeros)])

        return {
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
            "n_zeros": num_zeros,
            "n_total": total_count,
            "sparsity": float(num_zeros / total_count),
        }

    # ---------- public API expected by the components ----------------------
    def get_all_feature_ids(self) -> list[int]:
        return sorted(self.neuron_dirs.keys())

    # Histogram – identical shape to the DuckDB version
    def get_feature_histogram(self, nid: int) -> pl.DataFrame:
        return self._histogram_for(nid)

    # Top activations (optionally limited & already sorted high→low)
    def get_feature_activations(
        self, nid: int, limit: int | None = None
    ) -> pl.DataFrame:
        df = self._load_single_neuron_df(nid)
        df = df.sort("activation", descending=True)
        if limit:
            df = df.head(limit)
        return df

    def get_feature_stats(
        self, nid: int, with_zeros: bool = False
    ) -> dict[str, float]:
        return self._stats_for(nid, with_zeros)

    # ----- token / pair / triple examples ----------------------------------
    def _read_example_csv(
        self, nid: int, fname: str, limit: int | None = None
    ) -> pd.DataFrame:
        p = self.neuron_dirs[nid] / fname
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        if limit:
            df = df.head(limit)
        return df

    def get_single_token_examples(
        self, nid: int, limit: int = 50
    ) -> pd.DataFrame:
        return self._read_example_csv(nid, "single_token_df.csv", limit)

    def get_top_examples(self, nid: int, limit: int = 50) -> pd.DataFrame:
        # Using "pair" as the canonical top-examples table per earlier code
        return self._read_example_csv(nid, "pair_df.csv", limit)

    def get_triple_examples(self, nid: int, limit: int = 50) -> pd.DataFrame:
        return self._read_example_csv(nid, "triple_df.csv", limit)
