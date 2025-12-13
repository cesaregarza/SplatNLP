"""
Configuration settings for feature analysis.

This module contains the configuration class for feature analysis.
"""

import os
from typing import Any, Optional

from splatnlp.feature_analysis.defaults import (
    DEFAULT_ANALYSIS_PARAMS,
    DEFAULT_DATA_PATHS,
    DEFAULT_DEVICE,
    DEFAULT_FEATURE_LABELS_PATH,
    DEFAULT_MODEL_PATHS,
    DEFAULT_PRIMARY_MODEL_PARAMS,
    DEFAULT_SAE_PARAMS,
)


class FeatureAnalysisConfig:
    """Configuration class for feature analysis settings."""

    def __init__(
        self,
        model_paths: Optional[dict[str, str]] = None,
        data_paths: Optional[dict[str, str]] = None,
        primary_model_params: Optional[dict[str, Any]] = None,
        sae_params: Optional[dict[str, Any]] = None,
        analysis_params: Optional[dict[str, Any]] = None,
        device: str = DEFAULT_DEVICE,
        feature_labels_path: str = DEFAULT_FEATURE_LABELS_PATH,
    ):
        """Initialize configuration with custom or default settings."""

        # Model paths
        self.model_paths = model_paths or DEFAULT_MODEL_PATHS.copy()

        # Data paths
        self.data_paths = data_paths or DEFAULT_DATA_PATHS.copy()

        # Model parameters
        self.primary_model_params = (
            primary_model_params or DEFAULT_PRIMARY_MODEL_PARAMS.copy()
        )
        self.sae_params = sae_params or DEFAULT_SAE_PARAMS.copy()

        # Analysis parameters
        self.analysis_params = analysis_params or DEFAULT_ANALYSIS_PARAMS.copy()

        # Device and paths
        self.device = device
        self.feature_labels_path = feature_labels_path

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self):
        """Validate that critical paths exist."""
        critical_paths = [
            self.model_paths.get("primary_model"),
            self.model_paths.get("sae_model"),
            self.model_paths.get("vocab"),
            self.model_paths.get("weapon_vocab"),
        ]

        missing_paths = []
        for path in critical_paths:
            if path and not os.path.exists(path):
                missing_paths.append(path)

        if missing_paths:
            print(f"Warning: Missing paths detected: {missing_paths}")

    def get_model_path(self, model_name: str) -> str:
        """Get path for a specific model."""
        return self.model_paths.get(model_name, "")

    def get_data_path(self, data_name: str) -> str:
        """Get path for a specific data file."""
        return self.data_paths.get(data_name, "")

    def get_analysis_param(self, param_name: str) -> Any:
        """Get a specific analysis parameter."""
        return self.analysis_params.get(param_name)

    def update_model_paths(self, new_paths: dict[str, str]):
        """Update model paths."""
        self.model_paths.update(new_paths)
        self._validate_paths()

    def update_data_paths(self, new_paths: dict[str, str]):
        """Update data paths."""
        self.data_paths.update(new_paths)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_paths": self.model_paths,
            "data_paths": self.data_paths,
            "primary_model_params": self.primary_model_params,
            "sae_params": self.sae_params,
            "analysis_params": self.analysis_params,
            "device": self.device,
            "feature_labels_path": self.feature_labels_path,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "FeatureAnalysisConfig":
        """Create configuration from dictionary."""
        return cls(
            model_paths=config_dict.get("model_paths"),
            data_paths=config_dict.get("data_paths"),
            primary_model_params=config_dict.get("primary_model_params"),
            sae_params=config_dict.get("sae_params"),
            analysis_params=config_dict.get("analysis_params"),
            device=config_dict.get("device", DEFAULT_DEVICE),
            feature_labels_path=config_dict.get(
                "feature_labels_path", DEFAULT_FEATURE_LABELS_PATH
            ),
        )

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "FeatureAnalysisConfig":
        """Load configuration from JSON file."""
        import json

        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration instance
default_config = FeatureAnalysisConfig()
