"""SAE feature analysis utilities."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_top_activating_examples(
    feature_idx: int,
    records_df: pd.DataFrame,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Return the `n_top` rows that maximally activate a given SAE feature.

    Args:
        feature_idx: Index of the SAE feature to analyze
        records_df: DataFrame containing SAE activations and metadata
        n_top: Number of top examples to return

    Returns:
        DataFrame slice with top activating examples and their activation values
    """
    if records_df.empty:
        raise ValueError("records_df is empty!")
    if feature_idx >= len(records_df.iloc[0]["sae_hidden"]):
        raise ValueError(f"Feature index {feature_idx} out of range.")

    activations = records_df["sae_hidden"].apply(
        lambda vec: float(vec[feature_idx])
    )
    top_idx = activations.nlargest(n_top).index
    return records_df.loc[top_idx].assign(
        feature_activation=activations.loc[top_idx]
    )


def format_example(
    row: pd.Series,
    feature_idx: int,
    inv_vocab_map: Dict[int, str],
    inv_weapon_vocab_map: Dict[int, str],
    weapon_data_map: Optional[Dict[str, Dict]] = None,
) -> str:
    """
    Pretty-print one row from records_df.

    Args:
        row: DataFrame row containing example data
        feature_idx: Index of the SAE feature being analyzed
        inv_vocab_map: Mapping from token IDs to ability names
        inv_weapon_vocab_map: Mapping from token IDs to weapon names
        weapon_data_map: Optional mapping from weapon names to their metadata

    Returns:
        Formatted string representation of the example
    """
    activation = row["feature_activation"]
    ability_tokens = row["ability_input_tokens"]
    weapon_id = row["weapon_id_token"]
    model_logits = row["model_logits"]
    top_logits = model_logits.argsort()[-10:]
    top_logits_names = [inv_vocab_map.get(tok) for tok in top_logits]

    # Decode abilities
    ability_names = [
        inv_vocab_map.get(tok, f"UNK_{tok}")
        for tok in ability_tokens
        if inv_vocab_map.get(tok) not in ("<PAD>", "<NULL>")
    ]

    # Decode weapon
    token_str = inv_weapon_vocab_map.get(weapon_id, f"weapon_id_{weapon_id}")
    weapon_name = token_str.split("_")[-1]
    weapon_data = (
        weapon_data_map.get(weapon_name, {}) if weapon_data_map else {}
    )

    return (
        f"  Activation: {activation:.4f}\n"
        f"  Weapon: {weapon_name} (ID: {weapon_id})\n"
        f"  Weapon Data: {weapon_data}\n"
        f"  Abilities (input): {', '.join(ability_names)}\n"
        f"  Abilities (output): {', '.join(top_logits_names)}\n"
    )


def get_examples_str(
    top_examples_df: pd.DataFrame,
    feature_idx: int,
    inv_vocab_map: Dict[int, str],
    inv_weapon_vocab_map: Dict[int, str],
    weapon_data_map: Optional[Dict[str, Dict]] = None,
) -> str:
    """
    Format multiple examples into a string.

    Args:
        top_examples_df: DataFrame with top activating examples
        feature_idx: Index of the SAE feature being analyzed
        inv_vocab_map: Mapping from token IDs to ability names
        inv_weapon_vocab_map: Mapping from token IDs to weapon names
        weapon_data_map: Optional mapping from weapon names to their metadata

    Returns:
        Formatted string with all examples
    """
    out = f"--- Top {len(top_examples_df)} examples for feature {feature_idx} ---\n"
    for i, row in top_examples_df.iterrows():
        out += f"Example {i}:\n"
        out += format_example(
            row,
            feature_idx,
            inv_vocab_map,
            inv_weapon_vocab_map,
            weapon_data_map,
        )
    return out


def rank_features_by_sparsity(
    sae_hidden_activations: np.ndarray,
    activation_thresh: float = 1e-2,
    min_frac_active: float = 0.01,
    max_frac_active: float = 0.30,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Rank SAE features by sparsity-aware magnitude.

    Args:
        sae_hidden_activations: Matrix of SAE hidden activations [N, D]
        activation_thresh: Threshold for considering a feature active
        min_frac_active: Minimum fraction of examples where feature is active
        max_frac_active: Maximum fraction of examples where feature is active
        top_k: Number of top features to return

    Returns:
        DataFrame with feature statistics and rankings
    """
    # Calculate basic statistics
    active_mask = np.abs(sae_hidden_activations) > activation_thresh
    frac_active = active_mask.mean(axis=0)
    mean_when_active = np.where(
        frac_active > 0,
        sae_hidden_activations.sum(axis=0) / (active_mask.sum(axis=0) + 1e-8),
        0.0,
    )

    # Score features: high magnitude + sparse
    score = mean_when_active / np.sqrt(frac_active + 1e-8)

    # Wrap in DataFrame
    stats = pd.DataFrame(
        {
            "feature": np.arange(sae_hidden_activations.shape[1]),
            "frac_active": frac_active,
            "mean_when_active": mean_when_active,
            "score": score,
        }
    ).sort_values("score", ascending=False)

    # Filter by sparsity
    mask = (stats["frac_active"] > min_frac_active) & (
        stats["frac_active"] < max_frac_active
    )
    stats_filtered = stats[mask]

    return stats_filtered.head(top_k)
