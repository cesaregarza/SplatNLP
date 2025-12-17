"""
Storage backend implementations for loading SAE feature activations.

Provides a protocol and concrete implementations for different storage formats:
- TransposedZarrLoader: O(1) feature access from [features x samples] zarr
- NeuronActsLoader: Sparse per-feature .npy files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ActivationLoader(Protocol):
    """Protocol for loading raw activation data for a feature."""

    def load(self, feature_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Load activations for a feature.

        Returns:
            Tuple of (indices, activations) arrays, or None if unavailable.
            - indices: int64 array of global sample indices
            - activations: float32 array of activation values
        """
        ...

    @property
    def is_available(self) -> bool:
        """Whether this loader is available (paths exist)."""
        ...


class TransposedZarrLoader:
    """Loads activations from transposed zarr format [features x samples]."""

    def __init__(self, zarr_path: Path | None):
        self._zarr = None
        self._path = zarr_path
        if zarr_path and zarr_path.exists():
            try:
                import zarr

                self._zarr = zarr.open_array(str(zarr_path), mode="r")
                logger.info(f"Loaded transposed zarr: {self._zarr.shape}")
            except Exception as e:
                logger.warning(f"Failed to load transposed zarr: {e}")
                self._zarr = None

    @property
    def is_available(self) -> bool:
        return self._zarr is not None

    def load(self, feature_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.is_available:
            return None

        feature_acts = self._zarr[feature_id, :]
        mask = feature_acts != 0

        if not mask.any():
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        indices = np.where(mask)[0].astype(np.int64)
        activations = feature_acts[mask].astype(np.float32)
        return indices, activations


class NeuronActsLoader:
    """Loads activations from per-feature sparse .npy files.

    Expected directory structure:
        neuron_acts_dir/
            neuron_0000/
                acts.npy  # float32 activation values
                idxs.npy  # int64 sample indices
            neuron_0001/
                ...
    """

    def __init__(self, neuron_acts_dir: Path | None):
        self._dir = neuron_acts_dir
        if neuron_acts_dir and neuron_acts_dir.exists():
            logger.info(f"Found neuron_acts format at {neuron_acts_dir}")

    @property
    def is_available(self) -> bool:
        return self._dir is not None and self._dir.exists()

    def load(self, feature_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.is_available:
            return None

        feature_dir = self._dir / f"neuron_{feature_id:04d}"
        acts_path = feature_dir / "acts.npy"
        idxs_path = feature_dir / "idxs.npy"

        if not acts_path.exists() or not idxs_path.exists():
            return None

        activations = np.load(acts_path).astype(np.float32)
        indices = np.load(idxs_path).astype(np.int64)
        return indices, activations
