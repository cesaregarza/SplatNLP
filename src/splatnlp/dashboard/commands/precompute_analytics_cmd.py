"""Command for precomputing all dashboard analytics for faster loading and interaction."""

import argparse
import gc
import logging
# import os # Not directly used in the functions being moved
from pathlib import Path
from typing import Dict, List, Tuple, Any # Ensure all necessary types are imported

import h5py
import joblib # For loading metadata and saving analytics
import numpy as np
import pandas as pd # For metadata handling
import torch
from tqdm import tqdm

# Model imports (assuming these paths are correct relative to where this command might be run from)
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logger = logging.getLogger(__name__)

# Helper functions (copied from the original script)

def compute_neuron_statistics(acts: np.ndarray) -> Dict[str, Any]:
    """Compute basic statistics for a neuron's activations."""
    if acts.size == 0: # Handle empty activations
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'q25': 0.0, 'q75': 0.0,
            'n_zeros': 0, 'n_total': 0, 'sparsity': 1.0,
            'histogram': {'counts': [], 'bin_edges': []}
        }

    percentiles = np.percentile(acts, [25, 50, 75])
    n_zeros = int(np.sum(acts == 0))
    n_total = len(acts)
    sparsity = float(n_zeros / n_total) if n_total > 0 else 1.0

    stats = {
        'mean': float(acts.mean()),
        'std': float(acts.std()),
        'min': float(acts.min()),
        'max': float(acts.max()),
        'median': float(percentiles[1]),
        'q25': float(percentiles[0]),
        'q75': float(percentiles[2]),
        'n_zeros': n_zeros,
        'n_total': n_total,
        'sparsity': sparsity,
    }
    
    # Compute histogram, handle empty or single-value arrays for np.histogram
    if acts.size > 1 and not np.all(acts == acts[0]):
        hist, bin_edges = np.histogram(acts, bins=50)
        stats['histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    else: # Handle cases not suitable for histogram (empty, all same value)
        stats['histogram'] = {
            'counts': [acts.size] if acts.size > 0 else [], # Count of the single value or empty
            'bin_edges': [acts.min(), acts.max()] if acts.size > 0 else []
        }
            
    return stats

def compute_top_examples(
    acts: np.ndarray,
    metadata_list: List[Dict],
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Compute top activating examples with their metadata."""
    if acts.size == 0 or not metadata_list or top_k == 0:
        return []

    # Ensure top_k is not larger than available activations
    actual_top_k = min(top_k, len(acts))
    if actual_top_k == 0: return []


    # Get top-k efficiently using argpartition
    # For very small arrays, argpartition might behave unexpectedly if k is too large.
    if actual_top_k >= len(acts): # If k is effectively "all", just sort all
        top_k_indices = np.argsort(acts)[::-1]
    else: # Use argpartition for larger arrays where it's efficient
        top_k_indices_unsorted = np.argpartition(acts, -actual_top_k)[-actual_top_k:]
        top_k_indices = top_k_indices_unsorted[np.argsort(acts[top_k_indices_unsorted])[::-1]]
    
    top_examples_list = []
    for rank, original_idx in enumerate(top_k_indices, 1):
        if original_idx < len(metadata_list): # Check index bounds
            meta = metadata_list[original_idx]
            example_data = {
                'rank': rank,
                'weapon_name': meta.get('weapon_name', 'Unknown'),
                'input_abilities_str': meta.get('input_abilities_str', ''), # Ensure these keys exist
                'activation_value': float(acts[original_idx]),
                'top_predicted_abilities_str': meta.get('top_predicted_abilities_str', ''),
                'original_index': int(original_idx), # This is the index in the original full metadata_list
                 # Ensure 'token_projections' key exists or provide default
                'token_projections_tooltip_data': meta.get('token_projections', [])
            }
            top_examples_list.append(example_data)
        else:
            logger.warning(f"Metadata index {original_idx} out of bounds during top example computation.")
    
    return top_examples_list

def compute_interval_examples(
    acts: np.ndarray,
    metadata_list: List[Dict],
    n_intervals: int = 10,
    examples_per_interval: int = 5
) -> Dict[str, Dict[str, Any]]:
    """Compute representative examples for each activation interval."""
    if acts.size == 0 or not metadata_list or n_intervals == 0 or examples_per_interval == 0:
        return {}

    min_act, max_act = acts.min(), acts.max()
    if min_act == max_act: # Handle case where all activations are the same
        # Create one interval representing all examples
        bounds = np.array([min_act, min_act + 1e-6]) # Make a tiny interval
        n_intervals = 1 # Force one interval
    else:
        bounds = np.linspace(min_act, max_act, n_intervals + 1)
        # Ensure the last bound includes the max value
        bounds[-1] = max_act + 1e-6 
    
    intervals_data_map = {}
    for i in range(n_intervals):
        current_lower_bound = bounds[i]
        current_upper_bound = bounds[i+1]
        
        mask = (acts >= current_lower_bound) & (acts < current_upper_bound)
        interval_indices = np.where(mask)[0] # These are indices relative to `acts` array
        
        if len(interval_indices) == 0:
            # Still might want to represent empty intervals
            # intervals_data_map[f"interval_{i}"] = { ... 'count': 0, 'examples': [] ...}
            continue # Or skip if no examples
        
        # Sample representative examples
        if len(interval_indices) > examples_per_interval:
            np.random.seed(42)  # For reproducibility
            selected_indices_in_interval = np.random.choice(interval_indices, examples_per_interval, replace=False)
        else:
            selected_indices_in_interval = interval_indices
        
        examples_list = []
        for original_idx in selected_indices_in_interval: # original_idx is an index into `acts` and `metadata_list`
            if original_idx < len(metadata_list): # Check bounds
                meta = metadata_list[original_idx]
                example_data = {
                    'weapon_name': meta.get('weapon_name', 'Unknown'),
                    'input_abilities_str': meta.get('input_abilities_str', ''),
                    'activation_value': float(acts[original_idx]),
                    'top_predicted_abilities_str': meta.get('top_predicted_abilities_str', ''),
                    'token_projections_tooltip_data': meta.get('token_projections', [])
                }
                examples_list.append(example_data)
            else:
                logger.warning(f"Metadata index {original_idx} out of bounds during interval example computation.")

        interval_key_name = f"interval_{i}"
        intervals_data_map[interval_key_name] = {
            'bounds_str': f"[{current_lower_bound:.3f} - {current_upper_bound:.3f})",
            'count': int(len(interval_indices)),
            'representative_examples': examples_list
        }
    
    return intervals_data_map

def compute_logit_influences(
    sae_model: SparseAutoencoder,
    primary_model: SetCompletionModel,
    feature_id: int, # This is the neuron index
    vocab_list: List[str], # Pass vocab to map token indices to names
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute top positive and negative logit influences for a feature."""
    if feature_id >= sae_model.decoder.weight.shape[0]:
        logger.warning(f"Feature ID {feature_id} out of bounds for SAE decoder. Skipping logit influences.")
        return {'positive': [], 'negative': []}

    decoder_vector = sae_model.decoder.weight[feature_id].detach().cpu() # Move to CPU
    
    # Project through output layer (decoder of primary model)
    # primary_model.output_layer.weight is likely (output_dim, hidden_dim)
    # decoder_vector is (hidden_dim)
    # We want influence per output token, so (hidden_dim) @ (hidden_dim, output_dim) if output_layer.weight is W_out.T
    # Or (output_dim, hidden_dim) @ (hidden_dim) if output_layer.weight is W_out
    
    # Assuming primary_model.output_layer.weight is (vocab_size, hidden_dim)
    output_layer_weight = primary_model.output_layer.weight.detach().cpu() # Move to CPU

    with torch.no_grad():
        # logit_influences = torch.matmul(decoder_vector, output_layer_weight.T) # if decoder_vector is feature activation pattern
        # If decoder_vector is from SAE (input_dim for SAE = hidden_dim of primary model)
        # and SAE's decoder.weight is (num_features, hidden_dim), then decoder_vector is a row.
        # And primary_model.output_layer.weight is (vocab_size, hidden_dim)
        # So, influence = W_out @ feature_vector_from_SAE_decoder
        logit_influences = torch.matmul(output_layer_weight, decoder_vector) # Shape: (vocab_size)

    # Get top positive and negative influences
    # Ensure top_k is not larger than vocab_size
    actual_top_k = min(top_k, len(vocab_list))

    top_values, top_indices = torch.topk(logit_influences, actual_top_k)
    # For negative, negate values then find topk, then negate values back
    neg_top_values, neg_top_indices = torch.topk(-logit_influences, actual_top_k) 
    
    positive_influences = [
        {'token_name': vocab_list[idx.item()] if idx.item() < len(vocab_list) else f"Token {idx.item()}", 
         'influence': float(val.item())}
        for val, idx in zip(top_values, top_indices)
    ]
    negative_influences = [
        {'token_name': vocab_list[idx.item()] if idx.item() < len(vocab_list) else f"Token {idx.item()}", 
         'influence': float(-val.item())} # Negate value back
        for val, idx in zip(neg_top_values, neg_top_indices)
    ]
    
    return {'positive': positive_influences, 'negative': negative_influences}

def compute_correlations(
    current_neuron_acts: np.ndarray, # Activations for the current neuron (1D array)
    all_neurons_acts: np.ndarray,   # All neuron activations (2D array: examples x neurons)
    model_logits_all_examples: Optional[np.ndarray], # All model logits (2D array: examples x vocab_size)
    current_neuron_idx: int,
    vocab_list: List[str], # Pass vocab to map token indices to names
    top_k_features: int = 5,
    top_k_tokens: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute feature-to-feature and feature-to-logit correlations."""
    correlations_data = {
        'feature_to_feature': [],
        'feature_to_logit': []
    }

    if current_neuron_acts.size == 0: return correlations_data

    # Feature-to-feature correlations
    num_total_neurons = all_neurons_acts.shape[1]
    feature_corrs_list = []
    # Consider a subset for performance if too many features, e.g., random sample or first/last N
    # For now, iterate up to num_total_neurons, but limit how many are stored.
    
    candidate_other_neurons = [idx for idx in range(num_total_neurons) if idx != current_neuron_idx]
    # Could add more sophisticated selection of candidate_other_neurons if performance is an issue.

    for other_neuron_idx in candidate_other_neurons:
        other_neuron_acts = all_neurons_acts[:, other_neuron_idx]
        if other_neuron_acts.size == 0: continue
        try:
            # np.corrcoef returns a matrix, need the specific value
            # Also, handle cases with zero variance (np.corrcoef might return nan or raise warning)
            if np.std(current_neuron_acts) < 1e-6 or np.std(other_neuron_acts) < 1e-6:
                corr_val = 0.0 # Or np.nan, then filter nans later
            else:
                corr_val = np.corrcoef(current_neuron_acts, other_neuron_acts)[0, 1]
            
            if np.isfinite(corr_val): # Ensure it's not NaN or Inf
                feature_corrs_list.append({
                    'feature_id': int(other_neuron_idx), # The ID of the other neuron
                    'correlation': float(corr_val)
                })
        except Exception as e:
            logger.debug(f"Could not compute feat-feat correlation for {current_neuron_idx} and {other_neuron_idx}: {e}")
    
    # Sort by absolute correlation and get top k
    feature_corrs_list.sort(key=lambda x: abs(x['correlation']), reverse=True)
    correlations_data['feature_to_feature'] = feature_corrs_list[:top_k_features]
    
    # Feature-to-logit correlations (if model_logits_all_examples are provided)
    if model_logits_all_examples is not None and model_logits_all_examples.shape[0] == current_neuron_acts.shape[0]:
        num_tokens_in_vocab = model_logits_all_examples.shape[1]
        token_corrs_list = []
        for token_idx in range(num_tokens_in_vocab):
            logit_values_for_token = model_logits_all_examples[:, token_idx]
            if logit_values_for_token.size == 0: continue
            try:
                if np.std(current_neuron_acts) < 1e-6 or np.std(logit_values_for_token) < 1e-6:
                    corr_val = 0.0
                else:
                    corr_val = np.corrcoef(current_neuron_acts, logit_values_for_token)[0, 1]

                if np.isfinite(corr_val):
                    token_corrs_list.append({
                        'token_id': int(token_idx),
                        'token_name': vocab_list[token_idx] if token_idx < len(vocab_list) else f"Token {token_idx}",
                        'correlation': float(corr_val)
                    })
            except Exception as e:
                logger.debug(f"Could not compute feat-logit correlation for neuron {current_neuron_idx} and token {token_idx}: {e}")

        token_corrs_list.sort(key=lambda x: abs(x['correlation']), reverse=True)
        correlations_data['feature_to_logit'] = token_corrs_list[:top_k_tokens]
            
    return correlations_data

# Main command function
def precompute_analytics_command(args: argparse.Namespace):
    """Precompute all analytics for the dashboard."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting precomputation of dashboard analytics.")

    device = args.device if hasattr(args, 'device') else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Models ---
    logger.info("Loading models...")
    # Vocab loading (essential for model init and logit influence interpretation)
    try:
        with open(args.vocab_path, 'r') as f:
            vocab_map = json.load(f) # Assuming json, not orjson here for standard lib
        vocab_list = [""] * len(vocab_map) # Create list for index-based lookup
        for token, index in vocab_map.items():
            if index < len(vocab_list): vocab_list[index] = token
            else: logger.warning(f"Token index {index} out of bounds for vocab_list size {len(vocab_list)}")
        vocab_size = len(vocab_list)
        logger.info(f"Loaded vocab from {args.vocab_path}, size: {vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load vocab from {args.vocab_path}: {e}. Cannot proceed without vocab.")
        return

    try:
        with open(args.weapon_vocab_path, 'r') as f:
            weapon_vocab_map = json.load(f)
        weapon_vocab_size = len(weapon_vocab_map)
        logger.info(f"Loaded weapon vocab from {args.weapon_vocab_path}, size: {weapon_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load weapon_vocab from {args.weapon_vocab_path}: {e}. Cannot proceed.")
        return

    # Model architecture parameters (should ideally come from a config or args)
    # Using defaults from the original script for now.
    # These should match the models whose checkpoints are being loaded.
    primary_model_params = {
        'vocab_size': vocab_size, 'weapon_vocab_size': weapon_vocab_size,
        'embedding_dim': args.embedding_dim, 'hidden_dim': args.hidden_dim,
        'output_dim': vocab_size, 'num_layers': args.num_layers,
        'num_heads': args.num_heads, 'num_inducing_points': args.num_inducing_points,
        'use_layer_norm': True, 'dropout': 0.0,
        'pad_token_id': vocab_map.get("<PAD>", vocab_map.get(args.pad_token_name, 0)) # Get PAD ID
    }
    sae_model_params = {
        'input_dim': args.hidden_dim, # SAE input_dim is primary model's hidden_dim
        'expansion_factor': args.sae_expansion_factor
    }

    primary_model = SetCompletionModel(**primary_model_params)
    sae_model = SparseAutoencoder(**sae_model_params)
    
    try:
        primary_model.load_state_dict(torch.load(args.primary_model_checkpoint, map_location=device))
        sae_model.load_state_dict(torch.load(args.sae_model_checkpoint, map_location=device))
        primary_model.to(device).eval()
        sae_model.to(device).eval()
        logger.info("Models loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Error loading model checkpoints: {e}")
        return

    # --- Load Activations and Metadata ---
    logger.info("Loading activations and metadata...")
    try:
        with h5py.File(args.activations_h5, 'r') as hf:
            if 'activations' not in hf:
                 logger.error(f"'activations' dataset not found in {args.activations_h5}")
                 return
            # Load all activations into memory. If too large, this needs chunking.
            # For precompute, often implies reading all data.
            all_neuron_activations = hf['activations'][:] 
        
        # Metadata can be a .pkl (DataFrame or list of dicts) or .joblib
        # Original script used joblib.load. Let's try that first.
        metadata_records_loaded = joblib.load(args.metadata_file)
        
        # Ensure metadata_list is a list of dicts
        if isinstance(metadata_records_loaded, pd.DataFrame):
            metadata_list_for_processing = metadata_records_loaded.to_dict('records')
        elif isinstance(metadata_records_loaded, list):
            metadata_list_for_processing = metadata_records_loaded
        elif isinstance(metadata_records_loaded, dict) and 'analysis_df_records' in metadata_records_loaded: # From generate_activations output
            if isinstance(metadata_records_loaded['analysis_df_records'], pd.DataFrame):
                 metadata_list_for_processing = metadata_records_loaded['analysis_df_records'].to_dict('records')
            else: # Assuming it's already a list of dicts
                 metadata_list_for_processing = metadata_records_loaded['analysis_df_records']
        else:
            logger.error(f"Unsupported metadata format in {args.metadata_file}. Expected DataFrame, list of dicts, or dict from generate_activations.")
            return
        logger.info(f"Loaded {all_neuron_activations.shape[0]} examples, {all_neuron_activations.shape[1]} neurons. Metadata for {len(metadata_list_for_processing)} examples.")
        if all_neuron_activations.shape[0] != len(metadata_list_for_processing):
            logger.warning("Mismatch between number of activation examples and metadata records. Analytics might be partial or incorrect.")
            # Decide on a consistent number of examples, e.g., the minimum
            # num_examples_to_process = min(all_neuron_activations.shape[0], len(metadata_list_for_processing))
            # all_neuron_activations = all_neuron_activations[:num_examples_to_process]
            # metadata_list_for_processing = metadata_list_for_processing[:num_examples_to_process]
            # logger.info(f"Adjusted to process {num_examples_to_process} examples.")


    except Exception as e:
        logger.error(f"Error loading activations or metadata: {e}")
        return

    num_total_neurons = all_neuron_activations.shape[1]
    
    # --- Precompute Analytics ---
    all_features_analytics_data = {'features': []} # Changed from 'analytics' to 'all_features_analytics_data'
    
    # Load model logits if available in metadata (generated by generate_activations_cmd)
    model_logits_all_examples_data = None
    if metadata_list_for_processing and 'model_logits' in metadata_list_for_processing[0]:
        logger.info("Attempting to load model_logits from metadata...")
        try:
            # Ensure all records have model_logits and they have consistent shape
            # This can be memory intensive.
            first_logit_shape = np.array(metadata_list_for_processing[0]['model_logits']).shape
            if len(first_logit_shape) != 1: # Expect 1D array (logits for one example)
                 raise ValueError(f"Expected 1D array for model_logits, got shape {first_logit_shape}")

            logits_list = []
            for record in metadata_list_for_processing:
                logits_list.append(record['model_logits'])
            model_logits_all_examples_data = np.array(logits_list) # Stack to 2D array
            logger.info(f"Successfully loaded model_logits with shape {model_logits_all_examples_data.shape}")
        except Exception as e:
            logger.warning(f"Failed to load or stack model_logits from metadata: {e}. Logit correlations will be skipped.")
            model_logits_all_examples_data = None
    else:
        logger.info("model_logits not found in metadata. Logit correlations will be skipped.")

    logger.info(f"Starting analytics computation for {num_total_neurons} neurons...")
    for neuron_idx_iter in tqdm(range(num_total_neurons), desc="Computing neuron analytics"):
        current_neuron_activations = all_neuron_activations[:, neuron_idx_iter]
        
        single_neuron_analytics_data = {
            'id': neuron_idx_iter,
            'statistics': compute_neuron_statistics(current_neuron_activations),
            'top_activating_examples': compute_top_examples(current_neuron_activations, metadata_list_for_processing, args.top_k_examples),
            'subsampled_intervals_grid': compute_interval_examples(current_neuron_activations, metadata_list_for_processing, args.n_intervals, args.examples_per_interval),
            'top_logit_influences': compute_logit_influences(sae_model, primary_model, neuron_idx_iter, vocab_list, args.top_k_logit_tokens),
            'correlations': compute_correlations(
                current_neuron_activations, all_neuron_activations, model_logits_all_examples_data, 
                neuron_idx_iter, vocab_list, args.top_k_corr_features, args.top_k_corr_tokens
            )
        }
        all_features_analytics_data['features'].append(single_neuron_analytics_data)
        
        if neuron_idx_iter > 0 and neuron_idx_iter % 50 == 0: # Periodically collect garbage
            gc.collect()
            logger.debug(f"Collected garbage after processing neuron {neuron_idx_iter}")
    
    # --- Save Analytics ---
    output_file_path = Path(args.output_analytics_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving precomputed analytics to {output_file_path}...")
    try:
        joblib.dump(all_features_analytics_data, output_file_path, compress=3) # Add compression
        logger.info("Successfully saved analytics.")
    except Exception as e:
        logger.error(f"Failed to save analytics to {output_file_path}: {e}")

    logger.info("Precomputation of dashboard analytics finished.")

# Example CLI setup (to be integrated into a main CLI script later)
# def main_cli_example():
#     parser = argparse.ArgumentParser(description="Precompute dashboard analytics command.")
#     # Data paths
#     parser.add_argument("--activations-h5", type=str, required=True, help="Path to HDF5 file with activations.")
#     parser.add_argument("--metadata-file", type=str, required=True, help="Path to metadata file (e.g., .pkl or .joblib from generate_activations).")
#     parser.add_argument("--output-analytics-file", type=str, required=True, help="Output file to save precomputed analytics (e.g., .joblib).")

#     # Model paths and vocabs
#     parser.add_argument("--primary-model-checkpoint", type=str, required=True, help="Path to primary model checkpoint (.pt).")
#     parser.add_argument("--sae-model-checkpoint", type=str, required=True, help="Path to SAE model checkpoint (.pt).")
#     parser.add_argument("--vocab-path", type=str, required=True, help="Path to vocabulary JSON file.")
#     parser.add_argument("--weapon-vocab-path", type=str, required=True, help="Path to weapon vocabulary JSON file.")
#     parser.add_argument("--pad-token-name", type=str, default="<PAD>", help="Name of the padding token in the vocab.")


#     # Model architecture parameters (must match the loaded checkpoints)
#     parser.add_argument("--embedding-dim", type=int, default=32)
#     parser.add_argument("--hidden-dim", type=int, default=512)
#     parser.add_argument("--num-layers", type=int, default=3)
#     parser.add_argument("--num-heads", type=int, default=8)
#     parser.add_argument("--num-inducing-points", type=int, default=32)
#     parser.add_argument("--sae-expansion-factor", type=float, default=4.0)

#     # Analytics parameters
#     parser.add_argument("--top-k-examples", type=int, default=20, help="Number of top activating examples to store per feature.")
#     parser.add_argument("--n-intervals", type=int, default=10, help="Number of intervals for subsampled examples grid.")
#     parser.add_argument("--examples-per-interval", type=int, default=5, help="Number of examples to show per interval.")
#     parser.add_argument("--top-k-logit-tokens", type=int, default=5, help="Number of top/bottom tokens for logit influence.")
#     parser.add_argument("--top-k-corr-features", type=int, default=5, help="Number of top correlated features.")
#     parser.add_argument("--top-k-corr-tokens", type=int, default=10, help="Number of top correlated tokens (logits).")
    
#     parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Auto-detects if None.")

#     cli_args = parser.parse_args()
#     if cli_args.device is None: # Handle auto-detection if not specified via CLI
#         cli_args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#     precompute_analytics_command(cli_args)

# if __name__ == '__main__':
#     # main_cli_example() # For direct testing
#     pass
