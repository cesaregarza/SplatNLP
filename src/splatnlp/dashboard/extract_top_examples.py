#!/usr/bin/env python3
"""Parallel CPU-optimized extraction of top examples per activation range."""

import argparse
import gc
import orjson as json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def compute_activation_ranges(acts: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation range boundaries and bin indices with optimizations."""
    if len(acts) == 0:
        return np.array([0.0, 1.0]), np.array([])
    
    lo, hi = acts.min(), acts.max() + 1e-6
    bounds = np.linspace(lo, hi, n_bins + 1)
    
    # Use searchsorted instead of digitize (faster)
    bin_indices = np.searchsorted(bounds[:-1], acts, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    return bounds, bin_indices


def compute_neuron_statistics(acts: np.ndarray) -> Dict[str, Any]:
    """Optimized computation of neuron statistics."""
    # Single pass for percentiles
    percentiles = np.percentile(acts, [25, 50, 75])
    n_zeros = np.count_nonzero(acts == 0)
    
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
        'sparsity': float(n_zeros / len(acts)),
    }


def extract_top_examples_per_range(
    acts: np.ndarray,
    n_bins: int = 10,
    top_k: int = 1000,
    metadata_list: List[Dict] = None
) -> Dict[int, Dict[str, Any]]:
    """Extract top K examples for each activation range."""
    bounds, bin_indices = compute_activation_ranges(acts, n_bins)
    
    results = {}
    original_indices = np.arange(len(acts))
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = mask.sum()
        
        if bin_count == 0:
            results[str(bin_idx)] = {
                'bounds': (float(bounds[bin_idx]), float(bounds[bin_idx + 1])),
                'count': 0,
                'examples': []
            }
            continue
        
        bin_original_indices = original_indices[mask]
        bin_acts = acts[mask]
        
        # Get top-k efficiently
        k = min(top_k, len(bin_acts))
        if k < len(bin_acts):
            # Use argpartition for O(n) selection instead of O(n log n) sort
            top_k_local_indices = np.argpartition(bin_acts, -k)[-k:]
            # Only sort the top k
            top_k_local_indices = top_k_local_indices[np.argsort(bin_acts[top_k_local_indices])[::-1]]
        else:
            top_k_local_indices = np.argsort(bin_acts)[::-1]
        
        top_global_indices = bin_original_indices[top_k_local_indices]
        
        # Extract metadata efficiently
        top_examples = []
        for idx in top_global_indices:
            meta = metadata_list[idx]
            example_data = {
                'index': int(idx),
                'activation': float(acts[idx]),
                'metadata': {
                    'text': meta.get('text', ''),
                    'weapon_id': meta.get('weapon_id', -1),
                    'label': meta.get('label', '')
                }
            }
            top_examples.append(example_data)
        
        results[str(bin_idx)] = {
            'bounds': (float(bounds[bin_idx]), float(bounds[bin_idx + 1])),
            'count': int(bin_count),
            'examples': top_examples
        }
    
    return results


def process_single_neuron(args):
    """Process a single neuron - designed for multiprocessing."""
    neuron_idx, activations_file, n_bins, top_k_per_bin, output_dir, metadata_list = args
    
    try:
        # Each process opens its own HDF5 file handle
        with h5py.File(activations_file, 'r') as hf:
            acts = hf['activations'][:, neuron_idx]
        
        # Extract top examples per range
        range_data = extract_top_examples_per_range(
            acts, n_bins, top_k_per_bin, metadata_list
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
        with open(neuron_file, 'wb') as f:
            f.write(json.dumps(neuron_data))
        
        return neuron_idx, True
    except Exception as e:
        return neuron_idx, f"Error: {e}"


def find_low_activation_examples_optimized(
    acts_matrix: np.ndarray,
    n_neurons: int,
    n_bins: int = 10,
    bottom_bins: int = 2,
    max_examples: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized search for examples with all neurons having low activations."""
    n_examples = acts_matrix.shape[0]
    
    # Compute threshold for each neuron (bottom_bins percentile)
    threshold_percentile = (bottom_bins / n_bins) * 100
    thresholds = np.percentile(acts_matrix[:, :n_neurons], threshold_percentile, axis=0)
    
    # Vectorized check: which examples have ALL neurons below threshold
    low_mask = np.all(acts_matrix[:, :n_neurons] <= thresholds, axis=1)
    low_indices = np.where(low_mask)[0]
    
    # Sample if too many
    if len(low_indices) > max_examples:
        np.random.seed(42)
        low_indices = np.random.choice(low_indices, max_examples, replace=False)
    
    # Get activation stats for these examples
    max_acts = np.max(acts_matrix[low_indices, :n_neurons], axis=1)
    mean_acts = np.mean(acts_matrix[low_indices, :n_neurons], axis=1)
    
    return low_indices, max_acts, mean_acts


def process_activations_parallel(
    activations_file: Path,
    metadata_file: Path,
    output_dir: Path,
    n_bins: int = 10,
    top_k_per_bin: int = 1000,
    max_neurons: int = None,
    resume: bool = True,
    num_workers: int = None
):
    """Main processing function with parallel optimization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect optimal worker count
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)  # Leave 2 cores for system
    
    print(f"Using {num_workers} parallel workers")
    
    # Load metadata
    print("Loading metadata...")
    start_time = time.time()
    with open(metadata_file, 'rb') as f:
        metadata_records = pickle.load(f)
    
    # Convert to list for fast access
    if isinstance(metadata_records, pd.DataFrame):
        metadata_list = metadata_records.to_dict('records')
        metadata_df = metadata_records
    elif isinstance(metadata_records, list):
        metadata_list = metadata_records
        metadata_df = pd.DataFrame.from_records(metadata_records)
    else:
        raise ValueError("Unexpected metadata format")
    
    print(f"Metadata preparation took {time.time() - start_time:.2f}s")
    
    # Get dataset dimensions
    with h5py.File(activations_file, 'r') as hf:
        n_examples, n_neurons = hf['activations'].shape
    
    if max_neurons:
        n_neurons = min(n_neurons, max_neurons)
    
    print(f"Processing {n_neurons} neurons with {n_examples} examples")
    
    # Check for existing files
    processed_neurons = set()
    if resume:
        existing_files = list(output_dir.glob('neuron_*.json'))
        for f in existing_files:
            try:
                idx = int(f.stem.split('_')[1])
                if idx < n_neurons:  # Only count if within our range
                    processed_neurons.add(idx)
            except:
                continue
        
        if processed_neurons:
            print(f"Found {len(processed_neurons)} existing neuron files, resuming...")
    
    # Prepare work items
    neurons_to_process = [i for i in range(n_neurons) if i not in processed_neurons]
    
    if neurons_to_process:
        print(f"Processing {len(neurons_to_process)} neurons...")
        
        # Prepare arguments for multiprocessing
        process_args = [
            (neuron_idx, activations_file, n_bins, top_k_per_bin, output_dir, metadata_list)
            for neuron_idx in neurons_to_process
        ]
        
        # Process in parallel with progress bar
        successful = 0
        failed = []
        
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=len(neurons_to_process), desc="Processing neurons") as pbar:
                for neuron_idx, result in pool.imap_unordered(process_single_neuron, process_args):
                    if result is True:
                        successful += 1
                    else:
                        failed.append((neuron_idx, result))
                    pbar.update(1)
        
        print(f"Successfully processed {successful} neurons")
        if failed:
            print(f"Failed to process {len(failed)} neurons: {failed[:5]}...")  # Show first 5
    
    # Find low activation examples
    print("\nFinding low activation examples...")
    
    # Process in chunks to manage memory
    chunk_size = min(50000, n_examples)
    all_low_indices = []
    all_max_acts = []
    all_mean_acts = []
    
    with h5py.File(activations_file, 'r') as hf:
        acts_dataset = hf['activations']
        
        for chunk_start in tqdm(range(0, n_examples, chunk_size), 
                                desc="Processing chunks for low activations"):
            chunk_end = min(chunk_start + chunk_size, n_examples)
            
            # Load chunk
            chunk_acts = acts_dataset[chunk_start:chunk_end, :n_neurons]
            
            # Find low examples in chunk
            low_indices, max_acts, mean_acts = find_low_activation_examples_optimized(
                chunk_acts, n_neurons, n_bins, bottom_bins=2, max_examples=100
            )
            
            # Adjust indices to global
            low_indices += chunk_start
            
            all_low_indices.extend(low_indices)
            all_max_acts.extend(max_acts)
            all_mean_acts.extend(mean_acts)
            
            del chunk_acts
            gc.collect()
    
    # Limit total and create output
    if len(all_low_indices) > 1000:
        # Sample to get 1000
        sample_indices = np.random.choice(len(all_low_indices), 1000, replace=False)
        all_low_indices = [all_low_indices[i] for i in sample_indices]
        all_max_acts = [all_max_acts[i] for i in sample_indices]
        all_mean_acts = [all_mean_acts[i] for i in sample_indices]
    
    # Create low examples output
    low_examples = []
    for idx, max_act, mean_act in zip(all_low_indices, all_max_acts, all_mean_acts):
        meta = metadata_list[idx]
        example_data = {
            'index': int(idx),
            'max_activation': float(max_act),
            'mean_activation': float(mean_act),
            'metadata': {
                'text': meta.get('text', ''),
                'weapon_id': meta.get('weapon_id', -1),
                'label': meta.get('label', '')
            }
        }
        low_examples.append(example_data)
    
    # Save low activation examples
    low_examples_file = output_dir / 'low_activation_examples.json'
    with open(low_examples_file, 'wb') as f:
        f.write(json.dumps(low_examples))
    
    print(f"\nSaved {len(low_examples)} low activation examples")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel CPU-optimized extraction of top examples per activation range"
    )
    parser.add_argument("--activations", type=Path, required=True, help="Path to HDF5 file with activations")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to pickle file with metadata")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of activation ranges (default: 10)")
    parser.add_argument("--top-k", type=int, default=1000, help="Top examples per bin (default: 1000)")
    parser.add_argument("--max-neurons", type=int, default=None, help="Maximum neurons to process (for testing)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: auto)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing files")
    
    args = parser.parse_args()
    
    process_activations_parallel(
        args.activations,
        args.metadata,
        args.output_dir,
        args.n_bins,
        args.top_k,
        args.max_neurons,
        resume=not args.no_resume,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()