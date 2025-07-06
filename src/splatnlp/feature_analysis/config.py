"""
Configuration settings for feature analysis.

This module contains default configurations and constants used
throughout the feature analysis package.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Default paths
DEFAULT_FEATURE_LABELS_PATH = "src/splatnlp/dashboard/feature_labels.json"
DEFAULT_VOCAB_PATH = "saved_models/dataset_v0_2_full/vocab.json"
DEFAULT_WEAPON_VOCAB_PATH = "saved_models/dataset_v0_2_full/weapon_vocab.json"

# Default model paths
DEFAULT_MODEL_PATHS = {
    "primary_model": "saved_models/dataset_v0_2_full/model.pth",
    "sae_model": "saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_model_final.pth",
    "sae_config": "saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_run_config.json",
    "vocab": "saved_models/dataset_v0_2_full/vocab.json",
    "weapon_vocab": "saved_models/dataset_v0_2_full/weapon_vocab.json",
}

# Default data paths
DEFAULT_DATA_PATHS = {
    "tokenized_data": "/root/dev/SplatNLP/test_data/tokenized/tokenized_data.csv",
    "meta_path": "/mnt/e/activations2/outputs/",
    "neurons_root": "/mnt/e/activations2/outputs/neuron_acts",
}

# Default model parameters
DEFAULT_PRIMARY_MODEL_PARAMS = {
    "embedding_dim": 32,
    "hidden_dim": 512,
    "num_layers": 3,
    "num_heads": 8,
    "num_inducing": 32,
    "use_layer_norm": True,
    "dropout": 0.0,  # Set to 0 for eval
}

# Default SAE parameters
DEFAULT_SAE_PARAMS = {
    "input_dim": 512,  # Should match primary model's hidden_dim
    "expansion_factor": 4.0,
}

# Analysis parameters
DEFAULT_ANALYSIS_PARAMS = {
    "num_examples_to_inspect": 50000,
    "n_top_examples_per_feature": 10,
    "num_activation_buckets": 5,
    "tfidf_top_k": 20,
    "output_influences_limit": 10,
}

# Device configuration
DEFAULT_DEVICE = "cuda"

# Special tokens
SPECIAL_TOKENS = {"<PAD>", "<NULL>"}

# High AP pattern (for regex matching)
HIGH_AP_PATTERN = r"_(21|29|38|51|57)$"

# Feature categories
FEATURE_CATEGORIES = {
    "tactical": "Gameplay strategies and build patterns",
    "mechanical": "Specific ability patterns and relationships",
    "strategic": "High-level build types and substitutions",
    "unknown": "Uncategorized features",
}


class FeatureAnalysisConfig:
    """Configuration class for feature analysis settings."""

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        data_paths: Optional[Dict[str, str]] = None,
        primary_model_params: Optional[Dict[str, any]] = None,
        sae_params: Optional[Dict[str, any]] = None,
        analysis_params: Optional[Dict[str, any]] = None,
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

    def get_analysis_param(self, param_name: str) -> any:
        """Get a specific analysis parameter."""
        return self.analysis_params.get(param_name)

    def update_model_paths(self, new_paths: Dict[str, str]):
        """Update model paths."""
        self.model_paths.update(new_paths)
        self._validate_paths()

    def update_data_paths(self, new_paths: Dict[str, str]):
        """Update data paths."""
        self.data_paths.update(new_paths)

    def to_dict(self) -> Dict[str, any]:
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
    def from_dict(cls, config_dict: Dict[str, any]) -> "FeatureAnalysisConfig":
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
