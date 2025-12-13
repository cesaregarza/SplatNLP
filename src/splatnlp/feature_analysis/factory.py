"""
Factory module for creating feature analyzers.

This module provides convenient factory functions for creating
FeatureAnalyzer instances with pre-loaded models and configurations.
"""

import json
import logging
from typing import Any, Optional

import torch

from splatnlp.feature_analysis.analyzer import FeatureAnalyzer
from splatnlp.feature_analysis.config import (
    FeatureAnalysisConfig,
    default_config,
)

logger = logging.getLogger(__name__)


def load_vocabularies(
    config: FeatureAnalysisConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    """Load vocab and weapon vocab from files."""

    # Load vocabularies
    vocab_path = config.get_model_path("vocab")
    weapon_vocab_path = config.get_model_path("weapon_vocab")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    with open(weapon_vocab_path, "r") as f:
        weapon_vocab = json.load(f)

    logger.info(f"Loaded vocab with {len(vocab)} tokens")
    logger.info(f"Loaded weapon vocab with {len(weapon_vocab)} tokens")

    return vocab, weapon_vocab


def load_sae_config(config: FeatureAnalysisConfig) -> dict[str, Any]:
    """Load SAE configuration from file."""

    sae_config_path = config.get_model_path("sae_config")

    with open(sae_config_path, "r") as f:
        sae_config = json.load(f)

    logger.info(
        f"SAE Config - Input Dim: {sae_config.get('input_dim')}, "
        f"Expansion Factor: {sae_config.get('expansion_factor')}"
    )

    return sae_config


def load_primary_model(
    config: FeatureAnalysisConfig,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
):
    """Load and initialize the primary model."""

    # Import here to avoid circular imports
    from splatnlp.model.models import SetCompletionModel

    # Get model parameters
    params = config.primary_model_params

    # Initialize primary model
    primary_model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=len(vocab),  # Output vocab size
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        num_inducing_points=params["num_inducing"],
        use_layer_norm=params["use_layer_norm"],
        dropout=params["dropout"],
    )

    # Load primary model checkpoint
    primary_model_path = config.get_model_path("primary_model")
    primary_model.load_state_dict(
        torch.load(primary_model_path, map_location=config.device)
    )
    primary_model.to(config.device)
    primary_model.eval()

    logger.info("Primary model loaded and ready")

    return primary_model


def load_sae_model(config: FeatureAnalysisConfig, sae_config: dict[str, Any]):
    """Load and initialize the SAE model."""

    # Import here to avoid circular imports
    from splatnlp.monosemantic_sae.models import SparseAutoencoder

    # Get SAE parameters from config
    input_dim = sae_config.get("input_dim", config.sae_params["input_dim"])
    expansion_factor = sae_config.get(
        "expansion_factor", config.sae_params["expansion_factor"]
    )

    # Initialize SAE model
    sae_model = SparseAutoencoder(
        input_dim=input_dim, expansion_factor=expansion_factor
    )

    # Load SAE checkpoint
    sae_model_path = config.get_model_path("sae_model")
    sae_model.load_state_dict(
        torch.load(sae_model_path, map_location=config.device)
    )
    sae_model.to(config.device)
    sae_model.eval()

    logger.info("SAE model loaded and ready")

    return sae_model


def create_feature_analyzer(
    config: Optional[FeatureAnalysisConfig] = None,
    meta_path: Optional[str] = None,
    neurons_root: Optional[str] = None,
    device: Optional[str] = None,
    feature_labels_path: Optional[str] = None,
) -> FeatureAnalyzer:
    """
    Create a fully configured FeatureAnalyzer instance.

    Args:
        config: Configuration object. If None, uses default config.
        meta_path: Path to metadata directory (overrides config).
        neurons_root: Path to neurons root directory (overrides config).
        device: Device to use (overrides config).
        feature_labels_path: Path to feature labels file (overrides config).

    Returns:
        Configured FeatureAnalyzer instance.
    """

    # Use default config if none provided
    if config is None:
        config = default_config

    # Override paths if provided
    if meta_path is not None:
        config.data_paths["meta_path"] = meta_path
    if neurons_root is not None:
        config.data_paths["neurons_root"] = neurons_root
    if device is not None:
        config.device = device
    if feature_labels_path is not None:
        config.feature_labels_path = feature_labels_path

    logger.info(f"Creating feature analyzer with device: {config.device}")

    # Load vocabularies
    vocab, weapon_vocab = load_vocabularies(config)

    # Load SAE configuration
    sae_config = load_sae_config(config)

    # Load models
    primary_model = load_primary_model(config, vocab, weapon_vocab)
    sae_model = load_sae_model(config, sae_config)

    # Create feature analyzer
    analyzer = FeatureAnalyzer(
        primary_model=primary_model,
        sae_model=sae_model,
        vocab=vocab,
        weapon_vocab=weapon_vocab,
        feature_labels_path=config.feature_labels_path,
        meta_path=config.data_paths.get("meta_path"),
        neurons_root=config.data_paths.get("neurons_root"),
        device=config.device,
    )

    logger.info("Feature analyzer created successfully")

    return analyzer


def create_feature_analyzer_from_notebook_config(
    primary_model_checkpoint: str,
    sae_model_checkpoint: str,
    sae_config_path: str,
    vocab_path: str,
    weapon_vocab_path: str,
    meta_path: str,
    neurons_root: str,
    feature_labels_path: str = "src/splatnlp/dashboard/feature_labels.json",
    device: str = "cuda",
) -> FeatureAnalyzer:
    """
    Create a feature analyzer using notebook-style configuration.

    This function mimics the configuration style used in the original notebook
    for backward compatibility.

    Args:
        primary_model_checkpoint: Path to primary model checkpoint
        sae_model_checkpoint: Path to SAE model checkpoint
        sae_config_path: Path to SAE configuration file
        vocab_path: Path to vocabulary file
        weapon_vocab_path: Path to weapon vocabulary file
        meta_path: Path to metadata directory
        neurons_root: Path to neurons root directory
        feature_labels_path: Path to feature labels file
        device: Device to use

    Returns:
        Configured FeatureAnalyzer instance.
    """

    # Create custom config from notebook parameters
    config = FeatureAnalysisConfig(
        model_paths={
            "primary_model": primary_model_checkpoint,
            "sae_model": sae_model_checkpoint,
            "sae_config": sae_config_path,
            "vocab": vocab_path,
            "weapon_vocab": weapon_vocab_path,
        },
        data_paths={
            "meta_path": meta_path,
            "neurons_root": neurons_root,
        },
        device=device,
        feature_labels_path=feature_labels_path,
    )

    return create_feature_analyzer(config)


def create_quick_analyzer(
    sae_run_path: str = "run_20250429_023422",
    model_base_path: str = "saved_models/dataset_v0_2_full",
    meta_path: str = "/mnt/e/activations2/outputs/",
    neurons_root: str = "/mnt/e/activations2/outputs/neuron_acts",
    device: str = "cuda",
) -> FeatureAnalyzer:
    """
    Create a feature analyzer with commonly used paths.

    Args:
        sae_run_path: SAE run directory name
        model_base_path: Base path for model files
        meta_path: Path to metadata directory
        neurons_root: Path to neurons root directory
        device: Device to use

    Returns:
        Configured FeatureAnalyzer instance.
    """

    return create_feature_analyzer_from_notebook_config(
        primary_model_checkpoint=f"{model_base_path}/model.pth",
        sae_model_checkpoint=f"{model_base_path}/sae_runs/{sae_run_path}/sae_model_final.pth",
        sae_config_path=f"{model_base_path}/sae_runs/{sae_run_path}/sae_run_config.json",
        vocab_path=f"{model_base_path}/vocab.json",
        weapon_vocab_path=f"{model_base_path}/weapon_vocab.json",
        meta_path=meta_path,
        neurons_root=neurons_root,
        device=device,
    )
