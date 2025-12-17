"""
Configuration dataclass for activation storage paths.

Centralizes path management for different model types (Full vs Ultra)
and storage formats (batch zarr, transposed zarr, neuron_acts sparse).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class StorageConfig:
    """Configuration for activation storage paths."""

    model_type: Literal["full", "ultra"]
    data_dir: Path
    examples_dir: Path
    transposed_dir: Path | None = None
    neuron_acts_dir: Path | None = None

    @classmethod
    def for_full_model(
        cls,
        data_dir: str = "/mnt/e/activations_full_efficient",
        examples_dir: str = "/mnt/e/activations_full_efficient/examples",
        transposed_dir: str | None = "/mnt/e/activations_full_transposed",
        neuron_acts_dir: str | None = "/mnt/e/activations2/outputs/neuron_acts",
    ) -> StorageConfig:
        """Factory for Full model configuration."""
        return cls(
            model_type="full",
            data_dir=Path(data_dir),
            examples_dir=Path(examples_dir),
            transposed_dir=Path(transposed_dir) if transposed_dir else None,
            neuron_acts_dir=Path(neuron_acts_dir) if neuron_acts_dir else None,
        )

    @classmethod
    def for_ultra_model(
        cls,
        data_dir: str = "/mnt/e/activations_ultra_efficient",
        examples_dir: str = "/mnt/e/dashboard_examples_optimized",
        transposed_dir: str | None = "/mnt/e/activations_ultra_transposed",
    ) -> StorageConfig:
        """Factory for Ultra model configuration."""
        return cls(
            model_type="ultra",
            data_dir=Path(data_dir),
            examples_dir=Path(examples_dir),
            transposed_dir=Path(transposed_dir) if transposed_dir else None,
            neuron_acts_dir=None,  # Ultra doesn't use neuron_acts format
        )

    @classmethod
    def from_model_type(
        cls,
        model_type: Literal["full", "ultra"],
        **overrides,
    ) -> StorageConfig:
        """Factory that selects config based on model_type."""
        if model_type == "full":
            return cls.for_full_model(**overrides)
        else:
            return cls.for_ultra_model(**overrides)
