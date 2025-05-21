"""SAE visualization utilities."""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_l0_sparsity_distribution(
    sae_hidden_activations: np.ndarray,
    activation_thresh: float = 1e-6,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot histogram of L0 sparsity distribution across examples.

    Args:
        sae_hidden_activations: Matrix of SAE hidden activations [N, D]
        activation_thresh: Threshold for considering a feature active
        figsize: Figure size (width, height)
    """
    if sae_hidden_activations.size == 0:
        logger.warning("Skipping L0 plot as no activations were collected.")
        return

    logger.info("Calculating L0 norms...")
    # Calculate L0 norm for each example (count of non-zero activations)
    l0_norms = (np.abs(sae_hidden_activations) > activation_thresh).sum(axis=1)

    plt.figure(figsize=figsize)
    plt.hist(l0_norms, bins=50, alpha=0.7, color="blue")
    plt.xlabel("L0 Norm (Number of Active Features per Example)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of L0 Norms (Mean: {l0_norms.mean():.2f})")
    plt.grid(axis="y", alpha=0.5)
    plt.show()


def plot_feature_activation_distribution(
    sae_hidden_activations: np.ndarray,
    feature_idx: int,
    activation_thresh: float = 1e-6,
    figsize: Tuple[int, int] = (10, 6),
    use_log_scale: bool = False,
) -> None:
    """
    Plot histogram of activation values for a specific feature.

    Args:
        sae_hidden_activations: Matrix of SAE hidden activations [N, D]
        feature_idx: Index of feature to plot
        activation_thresh: Threshold for considering a feature active
        figsize: Figure size (width, height)
        use_log_scale: Whether to use log scale for y-axis
    """
    if sae_hidden_activations.size == 0:
        logger.warning(
            "Skipping activation plot as no activations were collected."
        )
        return

    logger.info(
        f"Plotting activation distribution for Feature {feature_idx}..."
    )
    feature_values = sae_hidden_activations[:, feature_idx]

    plt.figure(figsize=figsize)
    plt.hist(
        feature_values,
        bins=50,
        alpha=0.7,
        color="green",
        range=(0, feature_values.max()),
    )
    plt.xlabel(f"Activation Value for Feature {feature_idx}")
    plt.ylabel("Frequency")
    plt.title(f"Activation Distribution for Feature {feature_idx}")
    plt.grid(axis="y", alpha=0.5)
    if use_log_scale:
        plt.yscale("log")
    plt.show()
