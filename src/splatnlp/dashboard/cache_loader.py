"""Helper functions for loading activation caches in different formats."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import h5py
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_activation_cache(cache_path: Path) -> Dict[str, Any]:
    """Load activation cache, supporting both old and new formats.

    Args:
        cache_path: Path to the cache file

    Returns:
        Dictionary with keys:
        - analysis_df_records: DataFrame with metadata
        - all_sae_hidden_activations: numpy array of activations
        - metadata: Optional metadata dict
    """
    # Try to load as pickle first to check format
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        # Check if it's our new streaming format
        if (
            isinstance(data, dict)
            and data.get("_loader_version") == "streaming_v1"
        ):
            logger.info("Detected streaming cache format v1")
            return load_streaming_cache_v1(data, cache_path)
    except:
        pass

    # Try standard joblib load (old format)
    try:
        logger.info("Attempting to load as standard joblib cache")
        cache = joblib.load(cache_path)
        return cache
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        raise


def load_streaming_cache_v1(metadata: dict, cache_path: Path) -> Dict[str, Any]:
    """Load streaming cache format v1.

    This format stores activations in a separate HDF5 file and metadata in joblib.
    """
    metadata_path = Path(metadata["metadata_path"])
    h5_path = Path(metadata["h5_path"])

    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    meta_data = joblib.load(metadata_path)

    # Load activations from HDF5
    logger.info(f"Loading activations from {h5_path}")
    logger.info(f"Shape: {metadata['shape']}")

    # Check if we should load into memory or return a lazy accessor
    total_size_gb = (metadata["shape"][0] * metadata["shape"][1] * 4) / (
        1024**3
    )  # float32 = 4 bytes
    logger.info(f"Total activation size: {total_size_gb:.2f} GB")

    if total_size_gb > 4.0:  # If larger than 4GB, use lazy loading
        logger.warning(
            f"Activations are large ({total_size_gb:.2f} GB), using lazy loading"
        )
        # Return a proxy object that loads chunks on demand
        activations = H5LazyArray(h5_path, metadata["shape"])
    else:
        # Load into memory for smaller datasets
        logger.info("Loading activations into memory...")
        with h5py.File(h5_path, "r") as f:
            activations = f["activations"][:]
        logger.info("Activations loaded successfully")

    return {
        "analysis_df_records": meta_data["analysis_df_records"],
        "all_sae_hidden_activations": activations,
        "metadata": meta_data.get("metadata", {}),
        "activation_h5_path": str(
            h5_path
        ),  # Keep path for direct access if needed
    }


class H5LazyArray:
    """Lazy array accessor for HDF5 files that loads data on demand."""

    def __init__(self, h5_path: Path, shape: tuple):
        self.h5_path = h5_path
        self.shape = shape
        self.dtype = np.float32

    def __getitem__(self, key):
        """Load data on demand."""
        with h5py.File(self.h5_path, "r") as f:
            return f["activations"][key]

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    def __array__(self):
        """Convert to numpy array by loading all data."""
        logger.warning("Loading entire activation array into memory...")
        with h5py.File(self.h5_path, "r") as f:
            return f["activations"][:]


def load_filesystem_cache(neurons_root: Path) -> Dict[str, Any]:
    """Load activation cache from filesystem.

    Args:
        neurons_root: Path to the root directory containing neuron folders

    Returns:
        Dictionary containing activation data
    """
    logger.info(f"Loading filesystem cache from {neurons_root}")

    cache = {
        "activations": {},
        "metadata": {},
        "feature_stats": {},
        "feature_correlations": {},
    }

    # Load each neuron's data
    for neuron_dir in neurons_root.glob("neuron_*"):
        if not neuron_dir.is_dir():
            continue

        neuron_id = int(neuron_dir.name.split("_")[1])

        # Load activations
        activations_file = neuron_dir / "activations.npy"
        if activations_file.exists():
            cache["activations"][neuron_id] = np.load(str(activations_file))

        # Load metadata
        metadata_file = neuron_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                cache["metadata"][neuron_id] = json.load(f)

        # Load feature stats
        stats_file = neuron_dir / "feature_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                cache["feature_stats"][neuron_id] = json.load(f)

        # Load correlations
        corr_file = neuron_dir / "correlations.json"
        if corr_file.exists():
            with open(corr_file) as f:
                cache["feature_correlations"][neuron_id] = json.load(f)

    return cache


def load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load activation cache from the specified path.

    Args:
        cache_path: Path to the cache file or directory

    Returns:
        Dictionary containing activation data
    """
    if cache_path.is_dir():
        return load_filesystem_cache(cache_path)
    else:
        raise ValueError(f"Unsupported cache path: {cache_path}")
