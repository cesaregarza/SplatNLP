#!/usr/bin/env python3
"""Precompute all dashboard analytics for faster loading and interaction."""

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logger = logging.getLogger(__name__)

def compute_neuron_statistics(acts: np.ndarray) -> Dict[str, Any]:
    """Compute basic statistics for a neuron's activations."""
    # Compute all percentiles at once (faster)
    percentiles = np.percentile(acts, [25, 50, 75])
    
    # Basic stats only - skip expensive operations
    stats = {
        'mean': float(acts.mean()),
        'std': float(acts.std()),
        'min': float(acts.min()),
        'max': float(acts.max()),
        'median': float(percentiles[1]),
        'q25': float(percentiles[0]),
        'q75': float(percentiles[2]),
        'n_zeros': int(np.sum(acts == 0)),
        'n_total': len(acts),
        'sparsity': float(np.sum(acts == 0) / len(acts)),
    }
    
    # Compute histogram
    hist, bin_edges = np.histogram(acts, bins=50)
    stats['histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    return stats

def compute_top_examples(
    acts: np.ndarray,
    metadata_list: List[Dict],
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Compute top activating examples with their metadata."""
    # Get top-k efficiently using argpartition
    top_k_indices = np.argpartition(acts, -top_k)[-top_k:]
    # Sort just the top k
    top_k_indices = top_k_indices[np.argsort(acts[top_k_indices])[::-1]]
    
    top_examples = []
    for rank, idx in enumerate(top_k_indices, 1):
        meta = metadata_list[idx]
        example_data = {
            'rank': rank,
            'weapon_name': meta.get('weapon_name', 'Unknown'),
            'input_abilities_str': meta.get('input_abilities_str', ''),
            'activation_value': float(acts[idx]),
            'top_predicted_abilities_str': meta.get('top_predicted_abilities_str', ''),
            'original_index': int(idx),
            'token_projections_tooltip_data': meta.get('token_projections', [])
        }
        top_examples.append(example_data)
    
    return top_examples

def compute_interval_examples(
    acts: np.ndarray,
    metadata_list: List[Dict],
    n_intervals: int = 10,
    examples_per_interval: int = 5
) -> Dict[str, Dict[str, Any]]:
    """Compute representative examples for each activation interval."""
    # Compute interval boundaries
    min_act, max_act = acts.min(), acts.max()
    bounds = np.linspace(min_act, max_act, n_intervals + 1)
    
    intervals_data = {}
    for i in range(n_intervals):
        # Get examples in this interval
        mask = (acts >= bounds[i]) & (acts < bounds[i + 1])
        interval_indices = np.where(mask)[0]
        
        if len(interval_indices) == 0:
            continue
        
        # Sample representative examples
        if len(interval_indices) > examples_per_interval:
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(interval_indices, examples_per_interval, replace=False)
        else:
            selected_indices = interval_indices
        
        # Get example data
        examples = []
        for idx in selected_indices:
            meta = metadata_list[idx]
            example_data = {
                'weapon_name': meta.get('weapon_name', 'Unknown'),
                'input_abilities_str': meta.get('input_abilities_str', ''),
                'activation_value': float(acts[idx]),
                'top_predicted_abilities_str': meta.get('top_predicted_abilities_str', ''),
                'token_projections_tooltip_data': meta.get('token_projections', [])
            }
            examples.append(example_data)
        
        interval_key = f"interval_{i}"
        intervals_data[interval_key] = {
            'bounds_str': f"[{bounds[i]:.3f} - {bounds[i+1]:.3f})",
            'count': int(len(interval_indices)),
            'representative_examples': examples
        }
    
    return intervals_data

def compute_logit_influences(
    sae_model: SparseAutoencoder,
    primary_model: SetCompletionModel,
    feature_id: int,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute top positive and negative logit influences."""
    # Get feature's decoder vector
    decoder_vector = sae_model.decoder.weight[feature_id].detach()
    
    # Project through output layer
    with torch.no_grad():
        logit_influences = torch.matmul(decoder_vector, primary_model.output_layer.weight.T)
    
    # Get top positive and negative influences
    values, indices = torch.topk(logit_influences, top_k)
    neg_values, neg_indices = torch.topk(-logit_influences, top_k)
    
    positive = [
        {'token_name': f"Token {idx.item()}", 'influence': float(val.item())}
        for val, idx in zip(values, indices)
    ]
    negative = [
        {'token_name': f"Token {idx.item()}", 'influence': float(-val.item())}
        for val, idx in zip(neg_values, neg_indices)
    ]
    
    return {'positive': positive, 'negative': negative}

def compute_correlations(
    acts: np.ndarray,
    all_acts: np.ndarray,
    logits: np.ndarray,
    feature_id: int,
    top_k_features: int = 5,
    top_k_tokens: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute feature-to-feature and feature-to-logit correlations."""
    # Feature-to-feature correlations
    feature_corrs = []
    for other_id in range(all_acts.shape[1]):
        if other_id == feature_id:
            continue
        corr = np.corrcoef(acts, all_acts[:, other_id])[0, 1]
        feature_corrs.append({
            'feature_id': int(other_id),
            'correlation': float(corr)
        })
    
    # Sort and get top k
    feature_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    feature_corrs = feature_corrs[:top_k_features]
    
    # Feature-to-logit correlations
    token_corrs = []
    for token_id in range(logits.shape[1]):
        corr = np.corrcoef(acts, logits[:, token_id])[0, 1]
        token_corrs.append({
            'token_id': int(token_id),
            'correlation': float(corr)
        })
    
    # Sort and get top k
    token_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    token_corrs = token_corrs[:top_k_tokens]
    
    return {
        'feature_to_feature': feature_corrs,
        'feature_to_logit': token_corrs
    }

def precompute_analytics(
    activations_file: Path,
    metadata_file: Path,
    primary_model: SetCompletionModel,
    sae_model: SparseAutoencoder,
    output_file: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Precompute all analytics for the dashboard."""
    logger.info("Loading activations and metadata...")
    with h5py.File(activations_file, 'r') as hf:
        activations = hf['activations'][:]
    
    with open(metadata_file, 'rb') as f:
        metadata_records = joblib.load(f)
    
    # Convert metadata to list for fast access
    if isinstance(metadata_records, pd.DataFrame):
        metadata_list = metadata_records.to_dict('records')
    else:
        metadata_list = metadata_records
    
    n_neurons = activations.shape[1]
    logger.info(f"Processing {n_neurons} neurons...")
    
    # Initialize analytics dictionary
    analytics = {'features': []}
    
    # Check if logits are available in metadata
    logits = None
    if metadata_list and 'model_logits' in metadata_list[0]:
        logger.info("Loading model logits from metadata...")
        try:
            logits = np.stack([record['model_logits'] for record in metadata_list])
            logger.info(f"Loaded logits with shape {logits.shape}")
        except Exception as e:
            logger.warning(f"Failed to load logits from metadata: {e}")
            logits = None
    
    # Process each neuron
    for neuron_idx in tqdm(range(n_neurons), desc="Computing neuron analytics"):
        acts = activations[:, neuron_idx]
        
        # Compute all analytics for this neuron
        neuron_data = {
            'id': neuron_idx,
            'statistics': compute_neuron_statistics(acts),
            'top_activating_examples': compute_top_examples(acts, metadata_list),
            'subsampled_intervals_grid': compute_interval_examples(acts, metadata_list),
            'top_logit_influences': compute_logit_influences(sae_model, primary_model, neuron_idx),
        }
        
        # Only compute correlations if logits are available
        if logits is not None:
            neuron_data['correlations'] = compute_correlations(acts, activations, logits, neuron_idx)
        else:
            # Compute only feature-to-feature correlations
            neuron_data['correlations'] = {
                'feature_to_feature': [],
                'feature_to_logit': []
            }
            # Compute feature-to-feature correlations manually
            feature_corrs = []
            for other_id in range(min(100, activations.shape[1])):  # Limit to first 100 for speed
                if other_id == neuron_idx:
                    continue
                try:
                    corr = np.corrcoef(acts, activations[:, other_id])[0, 1]
                    if np.isfinite(corr):
                        feature_corrs.append({
                            'feature_id': int(other_id),
                            'correlation': float(corr)
                        })
                except:
                    pass
            feature_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            neuron_data['correlations']['feature_to_feature'] = feature_corrs[:5]
        
        analytics['features'].append(neuron_data)
        
        # Clear memory periodically
        if neuron_idx % 10 == 0:
            gc.collect()
    
    # Save analytics
    logger.info(f"Saving analytics to {output_file}...")
    joblib.dump(analytics, output_file)
    logger.info("Done!")

def main():
    parser = argparse.ArgumentParser(description="Precompute dashboard analytics")
    parser.add_argument("--activations", type=Path, required=True, help="Path to HDF5 file with activations")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata file")
    parser.add_argument("--primary-model", type=Path, required=True, help="Path to primary model checkpoint")
    parser.add_argument("--sae-model", type=Path, required=True, help="Path to SAE model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output file for analytics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load models
    logger.info("Loading models...")
    
    # Need to load vocab files to get model dimensions
    # For now, we'll use default dimensions - this should be improved
    # by either passing dimensions as arguments or loading from a config
    
    # Load vocab files to get correct dimensions
    import json
    
    # Try to find vocab files in common locations
    vocab_paths = [
        Path("saved_models/dataset_v0_2_full/vocab.json"),
        Path("vocab.json"),
        Path("test_data/tokenized/vocab.json")
    ]
    weapon_vocab_paths = [
        Path("saved_models/dataset_v0_2_full/weapon_vocab.json"),
        Path("weapon_vocab.json"),
        Path("test_data/tokenized/weapon_vocab.json")
    ]
    
    vocab_size = 140  # Default fallback
    weapon_vocab_size = 130  # Default fallback
    
    # Try to load vocab to get exact size
    for vpath in vocab_paths:
        if vpath.exists():
            with open(vpath) as f:
                vocab = json.load(f)
                vocab_size = len(vocab)
                logger.info(f"Loaded vocab from {vpath}, size: {vocab_size}")
                break
    
    for wpath in weapon_vocab_paths:
        if wpath.exists():
            with open(wpath) as f:
                weapon_vocab = json.load(f)
                weapon_vocab_size = len(weapon_vocab)
                logger.info(f"Loaded weapon vocab from {wpath}, size: {weapon_vocab_size}")
                break
    
    # Model architecture parameters
    embedding_dim = 32
    hidden_dim = 512
    num_layers = 3
    num_heads = 8
    num_inducing_points = 32
    sae_expansion_factor = 4.0
    
    # Initialize models with architecture
    primary_model = SetCompletionModel(
        vocab_size=vocab_size,
        weapon_vocab_size=weapon_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_inducing_points=num_inducing_points,
        use_layer_norm=True,
        dropout=0.0,
        pad_token_id=0  # Assuming 0 is pad token
    )
    
    sae_model = SparseAutoencoder(
        input_dim=hidden_dim,
        expansion_factor=sae_expansion_factor
    )
    
    # Load state dicts
    primary_model.load_state_dict(torch.load(args.primary_model, map_location=args.device))
    sae_model.load_state_dict(torch.load(args.sae_model, map_location=args.device))
    
    primary_model.to(args.device).eval()
    sae_model.to(args.device).eval()
    
    # Precompute analytics
    precompute_analytics(
        args.activations,
        args.metadata,
        primary_model,
        sae_model,
        args.output,
        args.device
    )

if __name__ == "__main__":
    main() 