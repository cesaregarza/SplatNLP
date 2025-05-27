"""Command for parallel CPU-optimized extraction of top examples per activation range."""

import argparse
import gc
import orjson as json # Changed from orjson to json to match usage
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import logging # Added logging

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
# from functools import partial # Not used

logger = logging.getLogger(__name__)


def compute_activation_ranges(acts: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation range boundaries and bin indices with optimizations."""
    if len(acts) == 0:
        # Return empty bounds and indices if acts is empty, ensuring bounds has 2 elements for consistency
        return np.array([0.0, 1.0]), np.array([]) 
    
    min_val, max_val = acts.min(), acts.max()
    # Ensure hi is slightly larger than max_val to include max_val in the last bin
    hi = max_val + 1e-6 if min_val != max_val else max_val + 1.0 # Handle case where all values are the same
    lo = min_val
    
    bounds = np.linspace(lo, hi, n_bins + 1)
    
    # Ensure bounds has at least two elements even for n_bins=0 or 1 (though n_bins is usually >=1)
    if len(bounds) < 2:
        bounds = np.array([lo, hi])

    # Use searchsorted instead of digitize (faster)
    # np.searchsorted requires bounds to be sorted, which linspace ensures.
    # We want bin_indices[i] = k if bounds[k] <= acts[i] < bounds[k+1]
    # np.searchsorted(bounds, x, side='right') gives index j such that all bounds[:j] < x <= all bounds[j:]
    # So, index is np.searchsorted(bounds, acts, side='right') - 1
    bin_indices = np.searchsorted(bounds, acts, side='right') -1
    
    # Clip to ensure indices are within [0, n_bins-1]
    # Max index should be n_bins-1 (for acts that fall into the last bin)
    # Min index should be 0
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    return bounds, bin_indices


def compute_neuron_statistics(acts: np.ndarray) -> Dict[str, Any]:
    """Optimized computation of neuron statistics."""
    if len(acts) == 0: # Handle empty array
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'q25': 0.0, 'q75': 0.0,
            'n_zeros': 0, 'n_total': 0, 'sparsity': 1.0, # Sparsity is 1 if no elements
        }

    # Single pass for percentiles
    percentiles = np.percentile(acts, [25, 50, 75])
    n_zeros = np.count_nonzero(acts == 0) # More direct way to count zeros
    
    return {
        'mean': float(acts.mean()),
        'std': float(acts.std()),
        'min': float(acts.min()),
        'max': float(acts.max()),
        'median': float(percentiles[1]),
        'q25': float(percentiles[0]),
        'q75': float(percentiles[2]),
        'n_zeros': int(n_zeros),
        'n_total': len(acts),
        'sparsity': float(n_zeros / len(acts)) if len(acts) > 0 else 1.0, # Avoid division by zero
    }


def extract_top_examples_per_range(
    acts: np.ndarray,
    n_bins: int = 10,
    top_k: int = 1000,
    metadata_list: List[Dict] = None # Should be List[Dict] as per usage
) -> Dict[str, Dict[str, Any]]: # Changed key type to str to match usage
    """Extract top K examples for each activation range."""
    if metadata_list is None: # Add default empty list for safety
        metadata_list = []

    bounds, bin_indices = compute_activation_ranges(acts, n_bins)
    
    results = {}
    original_indices = np.arange(len(acts)) # Efficiently get original indices
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = np.sum(mask) # More direct sum for boolean mask
        
        current_bin_bounds = (float(bounds[bin_idx]), float(bounds[bin_idx + 1])) if bin_idx < len(bounds) -1 else (float(bounds[bin_idx]), float(bounds[bin_idx]))


        if bin_count == 0:
            results[str(bin_idx)] = {
                'bounds': current_bin_bounds,
                'count': 0,
                'examples': []
            }
            continue
        
        bin_original_indices = original_indices[mask]
        bin_acts = acts[mask]
        
        # Get top-k efficiently
        # k should not exceed the number of available examples in the bin
        k = min(top_k, len(bin_acts)) 
        if k == 0 : # Handle case where k is 0 (e.g. top_k=0 or bin_acts is empty)
            top_k_local_indices = np.array([], dtype=int)
        elif k < len(bin_acts):
            # Use argpartition for O(n) selection instead of O(n log n) sort
            # We want largest k, so partition around -k and take elements from -k onwards
            top_k_local_indices = np.argpartition(bin_acts, -k)[-k:]
            # Only sort the top k selected elements (descending)
            top_k_local_indices = top_k_local_indices[np.argsort(bin_acts[top_k_local_indices])[::-1]]
        else: # k >= len(bin_acts), so take all elements sorted
            top_k_local_indices = np.argsort(bin_acts)[::-1]
        
        top_global_indices = bin_original_indices[top_k_local_indices]
        
        # Extract metadata efficiently
        top_examples = []
        for glob_idx in top_global_indices:
            # Check if glob_idx is within bounds of metadata_list
            if glob_idx < len(metadata_list):
                 meta = metadata_list[glob_idx]
                 example_data = {
                    'index': int(glob_idx), # Use global index
                    'activation': float(acts[glob_idx]), # Use global index for acts
                    'metadata': { # Ensure all keys are present, provide defaults
                        'text': meta.get('text', ''),
                        'weapon_id': meta.get('weapon_id', -1), # Use .get for safety
                        'label': meta.get('label', '')
                    }
                }
            else: # metadata_list might be shorter than acts array if not passed correctly
                example_data = {
                    'index': int(glob_idx),
                    'activation': float(acts[glob_idx]),
                     'metadata': { 'text': 'Error: metadata missing', 'weapon_id': -1, 'label': 'Error'}
                }

            top_examples.append(example_data)
        
        results[str(bin_idx)] = {
            'bounds': current_bin_bounds,
            'count': int(bin_count),
            'examples': top_examples
        }
    
    return results


# Global variable to store metadata_list for multiprocessing
# This avoids passing large metadata_list to each process
_metadata_list_global = None

def init_worker(metadata_list_for_worker: List[Dict]):
    """Initializer for multiprocessing workers to set global metadata."""
    global _metadata_list_global
    _metadata_list_global = metadata_list_for_worker


def process_single_neuron(args_tuple): # Renamed from 'args' to avoid conflict
    """Process a single neuron - designed for multiprocessing."""
    # Unpack arguments
    neuron_idx, activations_file_path_str, n_bins, top_k_per_bin, output_dir_path_str = args_tuple
    
    # Paths should be converted back to Path objects if needed, or use strings directly
    activations_file = Path(activations_file_path_str)
    output_dir = Path(output_dir_path_str)

    global _metadata_list_global # Use the global metadata

    try:
        # Each process opens its own HDF5 file handle
        with h5py.File(activations_file, 'r') as hf:
            if 'activations' not in hf:
                 return neuron_idx, "Error: 'activations' dataset not found in HDF5 file."
            if neuron_idx >= hf['activations'].shape[1]:
                return neuron_idx, f"Error: neuron_idx {neuron_idx} out of bounds."
            acts = hf['activations'][:, neuron_idx]
        
        # Extract top examples per range
        range_data = extract_top_examples_per_range(
            acts, n_bins, top_k_per_bin, _metadata_list_global # Use global metadata
        )
        
        # Compute statistics
        stats = compute_neuron_statistics(acts)
        
        # Prepare neuron data
        neuron_data = {
            'neuron_id': neuron_idx,
            'statistics': stats,
            'range_examples': range_data
        }
        
        # Save to file
        neuron_file = output_dir / f'neuron_{neuron_idx:05d}.json'
        with open(neuron_file, 'wb') as f: # Use 'wb' for orjson.dumps
            f.write(json.dumps(neuron_data)) # orjson.dumps returns bytes
        
        return neuron_idx, True # Return True for success
    except Exception as e:
        logger.error(f"Error processing neuron {neuron_idx}: {e}", exc_info=True) # Log full traceback
        return neuron_idx, f"Error: {e}"


def find_low_activation_examples_optimized(
    acts_matrix: np.ndarray, # This is a chunk of the full acts_matrix
    n_neurons_to_consider: int, # Number of neurons to use for thresholding (e.g. args.max_neurons)
    n_bins: int = 10,
    bottom_bins_threshold: int = 2, # Number of bins from bottom to consider "low"
    max_examples_to_return: int = 1000 # Max examples to return from this chunk
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized search for examples with all neurons having low activations within a chunk."""
    if acts_matrix.shape[0] == 0 or n_neurons_to_consider == 0: # Handle empty input
        return np.array([]), np.array([]), np.array([])

    # Consider only the specified number of neurons (up to max_neurons)
    relevant_acts_matrix = acts_matrix[:, :n_neurons_to_consider]

    # Compute threshold for each neuron (e.g., value at the top of the `bottom_bins_threshold`-th bin)
    threshold_percentile = (bottom_bins_threshold / n_bins) * 100
    
    # Ensure relevant_acts_matrix is not empty before percentile calculation
    if relevant_acts_matrix.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.percentile(relevant_acts_matrix, threshold_percentile, axis=0)
    
    # Vectorized check: which examples have ALL relevant neurons below their respective thresholds
    low_mask = np.all(relevant_acts_matrix <= thresholds, axis=1)
    low_indices_in_chunk = np.where(low_mask)[0] # Indices within the current chunk
    
    # Sample if too many examples found in this chunk
    if len(low_indices_in_chunk) > max_examples_to_return:
        # No need for np.random.seed here unless strict reproducibility per chunk is needed
        low_indices_in_chunk = np.random.choice(low_indices_in_chunk, max_examples_to_return, replace=False)
    
    # Get activation stats for these examples (max and mean over the relevant neurons)
    if len(low_indices_in_chunk) > 0: # Proceed only if low examples are found
        max_acts = np.max(relevant_acts_matrix[low_indices_in_chunk, :], axis=1)
        mean_acts = np.mean(relevant_acts_matrix[low_indices_in_chunk, :], axis=1)
    else:
        max_acts = np.array([])
        mean_acts = np.array([])

    return low_indices_in_chunk, max_acts, mean_acts


def extract_top_examples_command(args: argparse.Namespace):
    """Main command function for extracting top examples."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Executing extract_top_examples_command")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    activations_file = Path(args.activations)
    metadata_file = Path(args.metadata)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1)
    logger.info(f"Using {num_workers} parallel workers")
    
    logger.info("Loading metadata...")
    start_time = time.time()
    
    # Load metadata. This part is crucial and needs to be robust.
    try:
        with open(metadata_file, 'rb') as f:
            metadata_payload = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        return # Exit if metadata cannot be loaded

    # The structure of metadata_payload can vary based on how it was saved.
    # It might be a DataFrame, a list of dicts, or a dict containing these.
    # Assuming it might be a dict with 'analysis_df_records' (from generate_activations_cmd.py)
    # or directly a list or DataFrame.
    metadata_list_for_processing: List[Dict]
    if isinstance(metadata_payload, list):
        metadata_list_for_processing = metadata_payload
    elif isinstance(metadata_payload, pd.DataFrame):
        metadata_list_for_processing = metadata_payload.to_dict('records')
    elif isinstance(metadata_payload, dict) and 'analysis_df_records' in metadata_payload:
        if isinstance(metadata_payload['analysis_df_records'], pd.DataFrame):
            metadata_list_for_processing = metadata_payload['analysis_df_records'].to_dict('records')
        elif isinstance(metadata_payload['analysis_df_records'], list):
            metadata_list_for_processing = metadata_payload['analysis_df_records']
        else:
            logger.error("Metadata 'analysis_df_records' is not a DataFrame or list.")
            return
    else:
        logger.error(f"Unexpected metadata format in {metadata_file}. Expected list, DataFrame, or dict with 'analysis_df_records'.")
        return

    if not metadata_list_for_processing:
        logger.warning("Metadata list is empty. Some features might not work as expected.")
    
    logger.info(f"Metadata loading and preparation took {time.time() - start_time:.2f}s. Loaded {len(metadata_list_for_processing)} records.")
    
    with h5py.File(activations_file, 'r') as hf:
        if 'activations' not in hf:
            logger.error(f"'activations' dataset not found in {activations_file}")
            return
        n_examples_total, n_neurons_total = hf['activations'].shape
    
    n_neurons_to_process = n_neurons_total
    if args.max_neurons is not None:
        n_neurons_to_process = min(n_neurons_total, args.max_neurons)
    
    logger.info(f"Processing {n_neurons_to_process} neurons (out of {n_neurons_total} total) with {n_examples_total} examples.")
    
    processed_neurons_indices = set()
    if not args.no_resume: # resume is True by default
        existing_files = list(output_dir.glob('neuron_*.json'))
        for f_path in existing_files:
            try:
                idx = int(f_path.stem.split('_')[1])
                if idx < n_neurons_to_process:
                    processed_neurons_indices.add(idx)
            except (ValueError, IndexError): # Catch parsing errors
                logger.warning(f"Could not parse neuron index from filename: {f_path}")
        
        if processed_neurons_indices:
            logger.info(f"Found {len(processed_neurons_indices)} existing neuron files, resuming...")
    
    neurons_to_actually_process = [i for i in range(n_neurons_to_process) if i not in processed_neurons_indices]
    
    if neurons_to_actually_process:
        logger.info(f"Actually processing {len(neurons_to_actually_process)} new neurons...")
        
        # Prepare arguments for multiprocessing: Pass string paths for better pickling
        mp_args = [
            (neuron_idx, str(activations_file), args.n_bins, args.top_k, str(output_dir))
            for neuron_idx in neurons_to_actually_process
        ]
        
        successful_count = 0
        failed_neurons_info = []
        
        # Use initializer to pass metadata_list_for_processing
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(metadata_list_for_processing,)) as pool:
            with tqdm(total=len(neurons_to_actually_process), desc="Processing neurons") as pbar:
                for neuron_idx_processed, result_status in pool.imap_unordered(process_single_neuron, mp_args):
                    if result_status is True: # Explicitly check for True
                        successful_count += 1
                    else:
                        failed_neurons_info.append((neuron_idx_processed, result_status))
                    pbar.update(1)
        
        logger.info(f"Successfully processed {successful_count} neurons.")
        if failed_neurons_info:
            logger.warning(f"Failed to process {len(failed_neurons_info)} neurons. First 5 errors: {failed_neurons_info[:5]}")
    else:
        logger.info("No new neurons to process based on existing files.")

    logger.info("\nFinding low activation examples...")
    
    # Max examples to sample globally for low activation
    # This is different from max_examples_per_chunk for find_low_activation_examples_optimized
    global_max_low_examples = 1000 
    
    # Determine how many examples to aim for per chunk to not oversample too early
    # This is a heuristic: if we have N chunks, aim for global_max_low_examples / N from each.
    # But also don't try to get more than global_max_low_examples from a single chunk.
    num_chunks_for_low_examples = int(np.ceil(n_examples_total / args.low_act_chunk_size)) if args.low_act_chunk_size > 0 else 1
    examples_per_chunk_target = max(1, min(global_max_low_examples, int(np.ceil(global_max_low_examples / num_chunks_for_low_examples)) if num_chunks_for_low_examples > 0 else global_max_low_examples))


    all_low_indices_global = []
    all_max_acts_global = []
    all_mean_acts_global = []
    
    with h5py.File(activations_file, 'r') as hf:
        acts_dset = hf['activations']
        
        for chunk_start_idx in tqdm(range(0, n_examples_total, args.low_act_chunk_size), 
                                desc="Processing chunks for low activations"):
            chunk_end_idx = min(chunk_start_idx + args.low_act_chunk_size, n_examples_total)
            if chunk_start_idx >= chunk_end_idx: continue # Skip empty chunk

            current_chunk_acts = acts_dset[chunk_start_idx:chunk_end_idx, :n_neurons_to_process]
            
            # Find low examples in this chunk
            low_indices_in_chunk, max_acts_in_chunk, mean_acts_in_chunk = find_low_activation_examples_optimized(
                current_chunk_acts, 
                n_neurons_to_process, # Pass the number of neurons we are considering
                args.n_bins, 
                bottom_bins_threshold=args.low_act_bottom_bins, # Use arg
                max_examples_to_return=examples_per_chunk_target 
            )
            
            # Adjust indices to be global (relative to the full dataset)
            global_indices_for_chunk = low_indices_in_chunk + chunk_start_idx
            
            all_low_indices_global.extend(global_indices_for_chunk.tolist()) # Convert numpy arrays to lists
            all_max_acts_global.extend(max_acts_in_chunk.tolist())
            all_mean_acts_global.extend(mean_acts_in_chunk.tolist())
            
            del current_chunk_acts
            gc.collect()
    
    # If we collected more than global_max_low_examples, sample down
    if len(all_low_indices_global) > global_max_low_examples:
        logger.info(f"Found {len(all_low_indices_global)} total low activation candidates, sampling down to {global_max_low_examples}.")
        sample_indices = np.random.choice(len(all_low_indices_global), global_max_low_examples, replace=False)
        
        all_low_indices_global = [all_low_indices_global[i] for i in sample_indices]
        all_max_acts_global = [all_max_acts_global[i] for i in sample_indices]
        all_mean_acts_global = [all_mean_acts_global[i] for i in sample_indices]
    
    low_examples_output_list = []
    for glob_idx, max_act_val, mean_act_val in zip(all_low_indices_global, all_max_acts_global, all_mean_acts_global):
        if glob_idx < len(metadata_list_for_processing): # Check bounds
            meta = metadata_list_for_processing[glob_idx]
            example_data = {
                'index': int(glob_idx),
                'max_activation': float(max_act_val),
                'mean_activation': float(mean_act_val),
                'metadata': {
                    'text': meta.get('text', ''),
                    'weapon_id': meta.get('weapon_id', -1),
                    'label': meta.get('label', '')
                }
            }
            low_examples_output_list.append(example_data)
        else:
            logger.warning(f"Metadata index {glob_idx} out of bounds for low activation example.")

    low_examples_file_path = output_dir / 'low_activation_examples.json'
    with open(low_examples_file_path, 'wb') as f: # Use 'wb' for orjson
        f.write(json.dumps(low_examples_output_list))
    
    logger.info(f"\nSaved {len(low_examples_output_list)} low activation examples to {low_examples_file_path}")
    logger.info(f"All results saved to {output_dir}")

# Example of how this might be added to a CLI main later:
# def main_cli():
#     parser = argparse.ArgumentParser(description="Extract top examples command.")
#     parser.add_argument("--activations", type=str, required=True, help="Path to HDF5 file with activations")
#     parser.add_argument("--metadata", type=str, required=True, help="Path to pickle file with metadata")
#     parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
#     parser.add_argument("--n-bins", type=int, default=10, help="Number of activation ranges")
#     parser.add_argument("--top-k", type=int, default=1000, help="Top examples per bin")
#     parser.add_argument("--max-neurons", type=int, default=None, help="Maximum neurons to process")
#     parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers")
#     parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing files")
#     parser.add_argument("--low-act-chunk-size", type=int, default=50000, help="Chunk size for finding low activation examples")
#     parser.add_argument("--low-act-bottom-bins", type=int, default=2, help="Number of bottom bins to define 'low activation'")
#     cli_args = parser.parse_args()
#     extract_top_examples_command(cli_args)

# if __name__ == "__main__":
#     # This is for direct script execution if needed, but the command is designed to be called from a central CLI
#     # For direct testing, you'd need to simulate argparse.Namespace or create a simple parser here.
#     # Example:
#     # test_args = argparse.Namespace(
#     #     activations="path/to/activations.h5",
#     #     metadata="path/to/metadata.pkl",
#     #     output_dir="output/extract_top_examples",
#     #     n_bins=10,
#     #     top_k=100,
#     #     max_neurons=None, # Process all
#     #     num_workers=None, # Auto-detect
#     #     no_resume=False,
#     #     low_act_chunk_size=50000,
#     #     low_act_bottom_bins=2
#     # )
#     # extract_top_examples_command(test_args)
#     pass
