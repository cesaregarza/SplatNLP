#!/usr/bin/env python3
"""Ultra-fast correlation computation using inverted indices and approximate methods."""

import argparse
import gc
import json
import orjson
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import numba
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # For fast similarity search (optional, fallback to sklearn)


def build_inverted_index(neuron_data_dir: Path, n_neurons: int, top_n: int = 100) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Build inverted index: example_id -> set of neurons that have it in top examples.
    
    Returns:
        example_to_neurons: Dict mapping example_id to set of neuron_ids
        neuron_to_examples: Dict mapping neuron_id to set of example_ids
    """
    print("Building inverted index...")
    example_to_neurons = defaultdict(set)
    neuron_to_examples = defaultdict(set)
    
    for neuron_id in tqdm(range(n_neurons), desc="Indexing neurons"):
        neuron_file = neuron_data_dir / f'neuron_{neuron_id:05d}.json'
        
        if not neuron_file.exists():
            continue
        
        try:
            with open(neuron_file, 'rb') as f:
                content = f.read()
                if not content:
                    continue
                data = orjson.loads(content)
            
            # Collect indices from all ranges
            example_ids = set()
            for bin_data in data['range_examples'].values():
                for ex in bin_data['examples'][:top_n]:
                    example_id = ex['index']
                    example_ids.add(example_id)
                    example_to_neurons[example_id].add(neuron_id)
            
            neuron_to_examples[neuron_id] = example_ids
            
        except Exception as e:
            print(f"Warning: Error reading neuron {neuron_id}: {e}")
    
    return dict(example_to_neurons), dict(neuron_to_examples)


def find_candidate_pairs(
    neuron_to_examples: Dict[int, Set[int]], 
    min_overlap: int = 10,
    max_pairs_per_neuron: int = 100
) -> List[Tuple[int, int]]:
    """Find candidate neuron pairs that have sufficient overlap.
    
    This avoids checking all O(n²) pairs by only considering neurons that share examples.
    """
    print("Finding candidate pairs...")
    candidate_pairs = set()
    
    # Convert to list for efficient access
    neuron_ids = sorted(neuron_to_examples.keys())
    n_neurons = len(neuron_ids)
    
    # For each neuron, find others with significant overlap
    for i, neuron_i in enumerate(tqdm(neuron_ids, desc="Finding pairs")):
        examples_i = neuron_to_examples[neuron_i]
        if len(examples_i) < min_overlap:
            continue
        
        # Count overlaps with other neurons efficiently
        overlap_counts = defaultdict(int)
        for example_id in examples_i:
            # This is where we leverage the inverted index
            # We only check neurons that share this example
            for neuron_j in neuron_ids[i+1:]:
                if example_id in neuron_to_examples[neuron_j]:
                    overlap_counts[neuron_j] += 1
        
        # Get top overlapping neurons
        top_overlaps = sorted(
            [(n, count) for n, count in overlap_counts.items() if count >= min_overlap],
            key=lambda x: x[1],
            reverse=True
        )[:max_pairs_per_neuron]
        
        for neuron_j, _ in top_overlaps:
            if neuron_i < neuron_j:
                candidate_pairs.add((neuron_i, neuron_j))
    
    candidate_list = list(candidate_pairs)
    print(f"Found {len(candidate_list)} candidate pairs (out of {n_neurons * (n_neurons - 1) // 2} possible)")
    return candidate_list


@numba.jit(nopython=True)
def fast_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Pearson correlation using Numba."""
    n = len(x)
    if n < 3:
        return 0.0
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_yy = np.sum(y * y)
    sum_xy = np.sum(x * y)
    
    num = n * sum_xy - sum_x * sum_y
    den_x = n * sum_xx - sum_x * sum_x
    den_y = n * sum_yy - sum_y * sum_y
    
    if den_x <= 0 or den_y <= 0:
        return 0.0
    
    return num / np.sqrt(den_x * den_y)


def compute_correlations_for_pairs(
    candidate_pairs: List[Tuple[int, int]],
    activations_file: Path,
    neuron_to_examples: Dict[int, Set[int]],
    correlation_type: str = 'pearson',
    batch_size: int = 10000
) -> List[Dict]:
    """Compute correlations only for candidate pairs."""
    print(f"Computing correlations for {len(candidate_pairs)} pairs...")
    
    # Adaptive batch size - if we have few pairs, process them all at once
    if len(candidate_pairs) < batch_size * 2:
        batch_size = len(candidate_pairs)
    
    # Load activation data more efficiently
    with h5py.File(activations_file, 'r') as hf:
        acts_dataset = hf['activations']
        
        # Get all unique examples needed
        all_examples = set()
        for neuron_id, examples in neuron_to_examples.items():
            all_examples.update(examples)
        all_examples = sorted(all_examples)
        
        print(f"Loading activations for {len(all_examples)} unique examples...")
        
        # Create example index mapping
        example_to_idx = {ex: i for i, ex in enumerate(all_examples)}
        
        # Load activations in batches
        correlations = []
        
        # Process pairs in batches to optimize memory access
        n_batches = (len(candidate_pairs) + batch_size - 1) // batch_size
        
        if n_batches == 1:
            # Single batch - show progress for computing correlations
            print("Processing all pairs in a single batch...")
            
            # Get all neurons needed
            all_neurons = set()
            for i, j in candidate_pairs:
                all_neurons.add(i)
                all_neurons.add(j)
            all_neurons = sorted(all_neurons)
            
            # Get all examples needed
            all_batch_examples = set()
            for neuron_id in all_neurons:
                all_batch_examples.update(neuron_to_examples.get(neuron_id, []))
            all_batch_examples = sorted(all_batch_examples)
            
            if all_batch_examples:
                # Load all activations at once
                print(f"Loading activations for {len(all_batch_examples)} examples and {len(all_neurons)} neurons...")
                batch_acts = acts_dataset[all_batch_examples][:, all_neurons]
                
                # Create mappings
                neuron_to_idx = {n: i for i, n in enumerate(all_neurons)}
                example_to_local_idx = {ex: i for i, ex in enumerate(all_batch_examples)}
                
                # Process pairs with progress bar
                for neuron_i, neuron_j in tqdm(candidate_pairs, desc="Computing correlations"):
                    # Get common examples
                    examples_i = neuron_to_examples[neuron_i]
                    examples_j = neuron_to_examples[neuron_j]
                    common_examples = examples_i.intersection(examples_j)
                    
                    if len(common_examples) < 3:
                        continue
                    
                    # Get local indices
                    common_local_indices = [
                        example_to_local_idx[ex] for ex in common_examples 
                        if ex in example_to_local_idx
                    ]
                    
                    if not common_local_indices:
                        continue
                    
                    # Extract activations
                    local_i = neuron_to_idx[neuron_i]
                    local_j = neuron_to_idx[neuron_j]
                    
                    acts_i = batch_acts[common_local_indices, local_i]
                    acts_j = batch_acts[common_local_indices, local_j]
                    
                    # Compute correlation
                    try:
                        if correlation_type == 'pearson':
                            corr = fast_pearson_correlation(acts_i, acts_j)
                        else:  # spearman
                            corr, _ = spearmanr(acts_i, acts_j)
                        
                        if np.isfinite(corr) and abs(corr) > 0.01:
                            correlations.append({
                                'neuron_i': int(neuron_i),
                                'neuron_j': int(neuron_j),
                                'correlation': float(corr),
                                'n_common': len(common_examples)
                            })
                    except Exception:
                        pass
                
                del batch_acts
                gc.collect()
        
        else:
            # Multiple batches - show batch progress
            print(f"Processing {len(candidate_pairs)} pairs in {n_batches} batches of {batch_size}...")
            
            for batch_idx, batch_start in enumerate(tqdm(range(0, len(candidate_pairs), batch_size), 
                                                         desc="Processing batches", total=n_batches)):
                batch_end = min(batch_start + batch_size, len(candidate_pairs))
                batch_pairs = candidate_pairs[batch_start:batch_end]
                
                # Get unique neurons in this batch
                batch_neurons = set()
                for i, j in batch_pairs:
                    batch_neurons.add(i)
                    batch_neurons.add(j)
                batch_neurons = sorted(batch_neurons)
                
                # Get examples for batch neurons
                batch_examples = set()
                for neuron_id in batch_neurons:
                    batch_examples.update(neuron_to_examples.get(neuron_id, []))
                batch_examples = sorted(batch_examples)
                
                if not batch_examples:
                    continue
                
                # Load activations for batch
                batch_acts = acts_dataset[batch_examples][:, batch_neurons]
                
                # Create local mappings
                local_neuron_to_idx = {n: i for i, n in enumerate(batch_neurons)}
                local_example_to_idx = {ex: i for i, ex in enumerate(batch_examples)}
                
                # Compute correlations for batch
                for neuron_i, neuron_j in batch_pairs:
                    # Get common examples
                    examples_i = neuron_to_examples[neuron_i]
                    examples_j = neuron_to_examples[neuron_j]
                    common_examples = examples_i.intersection(examples_j)
                    
                    if len(common_examples) < 3:
                        continue
                    
                    # Get local indices
                    common_local_indices = [
                        local_example_to_idx[ex] for ex in common_examples 
                        if ex in local_example_to_idx
                    ]
                    
                    if not common_local_indices:
                        continue
                    
                    # Extract activations
                    local_i = local_neuron_to_idx[neuron_i]
                    local_j = local_neuron_to_idx[neuron_j]
                    
                    acts_i = batch_acts[common_local_indices, local_i]
                    acts_j = batch_acts[common_local_indices, local_j]
                    
                    # Compute correlation
                    try:
                        if correlation_type == 'pearson':
                            corr = fast_pearson_correlation(acts_i, acts_j)
                        else:  # spearman
                            corr, _ = spearmanr(acts_i, acts_j)
                        
                        if np.isfinite(corr) and abs(corr) > 0.01:
                            correlations.append({
                                'neuron_i': int(neuron_i),
                                'neuron_j': int(neuron_j),
                                'correlation': float(corr),
                                'n_common': len(common_examples)
                            })
                    except Exception:
                        pass
                
                # Clear memory
                del batch_acts
                gc.collect()
    
    return correlations


def compute_approximate_correlations(
    activations_file: Path,
    neuron_data_dir: Path,
    output_file: Path,
    n_neurons: Optional[int] = None,
    top_n_per_bin: int = 100,
    correlation_type: str = 'pearson',
    correlation_threshold: float = 0.3,
    min_overlap: int = 10,
    max_pairs_per_neuron: int = 100
):
    """Compute correlations using approximate methods for massive speedup."""
    
    start_time = time.time()
    
    # Determine number of neurons
    with h5py.File(activations_file, 'r') as hf:
        total_examples, total_neurons = hf['activations'].shape
        if n_neurons is None:
            n_neurons = total_neurons
        else:
            n_neurons = min(n_neurons, total_neurons)
    
    print(f"Processing {n_neurons} neurons")
    
    # Step 1: Build inverted index
    example_to_neurons, neuron_to_examples = build_inverted_index(
        neuron_data_dir, n_neurons, top_n_per_bin
    )
    
    print(f"Indexed {len(example_to_neurons)} unique examples across {len(neuron_to_examples)} neurons")
    
    # Step 2: Find candidate pairs using inverted index
    candidate_pairs = find_candidate_pairs(
        neuron_to_examples, min_overlap, max_pairs_per_neuron
    )
    
    # Step 3: Compute correlations only for candidate pairs
    correlations = compute_correlations_for_pairs(
        candidate_pairs, activations_file, neuron_to_examples, correlation_type
    )
    
    # Filter by threshold
    significant_correlations = [
        corr for corr in correlations
        if abs(corr['correlation']) > correlation_threshold
    ]
    
    print(f"\nFound {len(significant_correlations)} significant correlations (|r| > {correlation_threshold})")
    print(f"Total computation time: {time.time() - start_time:.1f} seconds")
    
    # Save results
    correlation_data = {
        'n_neurons': n_neurons,
        'correlation_type': correlation_type,
        'threshold': correlation_threshold,
        'min_overlap': min_overlap,
        'correlations': significant_correlations
    }
    
    with open(output_file, 'w') as f:
        json.dump(correlation_data, f, indent=2)
    
    # Save sparse matrix
    sparse_file = output_file.with_suffix('.npz')
    row = []
    col = []
    data = []
    
    for corr in significant_correlations:
        i, j = corr['neuron_i'], corr['neuron_j']
        val = corr['correlation']
        row.extend([i, j])
        col.extend([j, i])
        data.extend([val, val])
    
    corr_matrix = sparse.csr_matrix(
        (data, (row, col)), shape=(n_neurons, n_neurons)
    )
    sparse.save_npz(sparse_file, corr_matrix)
    
    print(f"Saved results to {output_file} and {sparse_file}")
    
    # Also save the candidate pairs for inspection
    stats = {
        'total_possible_pairs': n_neurons * (n_neurons - 1) // 2,
        'candidate_pairs_checked': len(candidate_pairs),
        'reduction_factor': (n_neurons * (n_neurons - 1) // 2) / max(len(candidate_pairs), 1),
        'significant_correlations': len(significant_correlations)
    }
    
    stats_file = output_file.with_suffix('.stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast correlation computation")
    
    parser.add_argument("--activations", type=Path, required=True, help="Path to HDF5 file with activations")
    parser.add_argument("--neuron-data-dir", type=Path, required=True, help="Directory with neuron data")
    parser.add_argument("--output", type=Path, required=True, help="Output file for correlations")
    parser.add_argument("--n-neurons", type=int, default=None, help="Number of neurons to process")
    parser.add_argument("--top-n-per-bin", type=int, default=100, help="Top examples per bin to use")
    parser.add_argument("--correlation-type", choices=['pearson', 'spearman'], default='pearson')
    parser.add_argument("--threshold", type=float, default=0.3, help="Correlation threshold")
    parser.add_argument("--min-overlap", type=int, default=10, help="Minimum overlap for correlation")
    parser.add_argument("--max-pairs-per-neuron", type=int, default=100, help="Max pairs to check per neuron")
    
    args = parser.parse_args()
    
    compute_approximate_correlations(
        args.activations,
        args.neuron_data_dir,
        args.output,
        args.n_neurons,
        args.top_n_per_bin,
        args.correlation_type,
        args.threshold,
        args.min_overlap,
        args.max_pairs_per_neuron
    )


if __name__ == "__main__":
    main()
