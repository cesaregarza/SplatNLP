"""Command for ultra-fast correlation computation using inverted indices and approximate methods."""

import argparse
import gc
import json # Standard json for output, orjson for loading if needed
import orjson # For loading neuron data
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time
from collections import defaultdict
import logging

import h5py
import numpy as np
# import pandas as pd # Not directly used in the core logic moved here
from scipy import sparse
# from scipy.spatial.distance import cosine # Not used
from scipy.stats import spearmanr # Pearson is custom
from tqdm import tqdm
import numba
# from sklearn.metrics.pairwise import cosine_similarity # Not used
# import faiss # Not used in the core logic moved here

logger = logging.getLogger(__name__)

def build_inverted_index(
    neuron_data_dir: Path, n_neurons: int, top_n: int = 100
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Build inverted index: example_id -> set of neurons that have it in top examples."""
    logger.info("Building inverted index...")
    example_to_neurons = defaultdict(set)
    neuron_to_examples = defaultdict(set)
    
    for neuron_id in tqdm(range(n_neurons), desc="Indexing neurons"):
        neuron_file = neuron_data_dir / f'neuron_{neuron_id:05d}.json'
        
        if not neuron_file.exists():
            logger.debug(f"Neuron file {neuron_file} not found, skipping.")
            continue
        
        try:
            with open(neuron_file, 'rb') as f:
                content = f.read()
                if not content:
                    logger.debug(f"Neuron file {neuron_file} is empty, skipping.")
                    continue
                data = orjson.loads(content) # Use orjson for speed
            
            current_neuron_examples = set()
            if 'range_examples' in data and isinstance(data['range_examples'], dict):
                for bin_data in data['range_examples'].values():
                    if 'examples' in bin_data and isinstance(bin_data['examples'], list):
                        for ex in bin_data['examples'][:top_n]: # Apply top_n limit here
                            if isinstance(ex, dict) and 'index' in ex:
                                example_id = ex['index']
                                current_neuron_examples.add(example_id)
                                example_to_neurons[example_id].add(neuron_id)
                            else:
                                logger.warning(f"Malformed example entry in {neuron_file}: {ex}")
                    else:
                        logger.warning(f"Missing or malformed 'examples' list in bin_data in {neuron_file}")
            else:
                logger.warning(f"Missing or malformed 'range_examples' in {neuron_file}")

            if current_neuron_examples: # Only add neuron if it has examples
                neuron_to_examples[neuron_id] = current_neuron_examples
            
        except orjson.JSONDecodeError as e:
            logger.warning(f"Warning: JSON decode error reading neuron {neuron_id} from {neuron_file}: {e}")
        except Exception as e:
            logger.warning(f"Warning: Generic error reading neuron {neuron_id} from {neuron_file}: {e}")
    
    logger.info(f"Built inverted index. example_to_neurons keys: {len(example_to_neurons)}, neuron_to_examples keys: {len(neuron_to_examples)}")
    return dict(example_to_neurons), dict(neuron_to_examples)


def find_candidate_pairs(
    neuron_to_examples: Dict[int, Set[int]], 
    min_overlap: int = 10,
    max_pairs_per_neuron: int = 100,
    example_to_neurons: Dict[int, Set[int]] = None # Pass this for efficiency
) -> List[Tuple[int, int]]:
    """Find candidate neuron pairs that have sufficient overlap."""
    logger.info("Finding candidate pairs...")
    candidate_pairs = set()
    
    neuron_ids = sorted(neuron_to_examples.keys()) # Neurons that actually have example data
    n_neurons_with_data = len(neuron_ids)
    
    # Precompute a map from example_id to neurons if not provided (though it should be from build_inverted_index)
    # This is a fallback or can be an alternative way if example_to_neurons is directly available and complete
    if example_to_neurons is None:
        logger.warning("example_to_neurons not provided to find_candidate_pairs, reconstructing. This might be slow.")
        example_to_neurons = defaultdict(set)
        for neuron_id, examples in neuron_to_examples.items():
            for ex_id in examples:
                example_to_neurons[ex_id].add(neuron_id)

    for i, neuron_i in enumerate(tqdm(neuron_ids, desc="Finding pairs")):
        examples_i = neuron_to_examples.get(neuron_i, set()) # Use .get for safety
        if len(examples_i) < min_overlap: # Skip if neuron_i itself has too few examples
            continue
        
        # Count overlaps with other neurons efficiently using the example_to_neurons map
        overlap_counts = defaultdict(int)
        for example_id in examples_i:
            # Neurons that also contain this example
            # Only consider neurons j > i to avoid duplicate pairs (j,i) and to avoid self-correlation (i,i)
            for neuron_j in example_to_neurons.get(example_id, set()):
                if neuron_j > neuron_i: # Ensure neuron_j is in our sorted list and after neuron_i
                    overlap_counts[neuron_j] += 1
        
        # Filter by min_overlap and select top N based on overlap count
        # Sort by overlap count descending, then by neuron_id ascending for stable sort
        valid_overlaps = [
            (neuron_j_id, count) for neuron_j_id, count in overlap_counts.items()
            if count >= min_overlap and neuron_j_id in neuron_to_examples # Ensure neuron_j_id is valid
        ]
        
        # Sort and limit
        # Sorting key: primary by count (desc), secondary by neuron_id (asc) for stability
        valid_overlaps.sort(key=lambda x: (-x[1], x[0]))
        
        for neuron_j_id, _ in valid_overlaps[:max_pairs_per_neuron]:
            # Pair is always (neuron_i, neuron_j_id) where neuron_i < neuron_j_id due to earlier check
            candidate_pairs.add((neuron_i, neuron_j_id))
            
    candidate_list = sorted(list(candidate_pairs)) # Sort for deterministic output
    logger.info(f"Found {len(candidate_list)} candidate pairs (out of {n_neurons_with_data * (n_neurons_with_data - 1) // 2} possible based on neurons with data)")
    return candidate_list


@numba.jit(nopython=True, cache=True) # Added cache=True
def fast_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Pearson correlation using Numba."""
    n = len(x)
    if n < 2: # Pearson needs at least 2 points; standard is 3 for reliability but 2 is min
        return 0.0 
    
    # Ensure inputs are float arrays for Numba operations
    x_float = x.astype(np.float64)
    y_float = y.astype(np.float64)

    sum_x = np.sum(x_float)
    sum_y = np.sum(y_float)
    sum_xx = np.sum(x_float * x_float)
    sum_yy = np.sum(y_float * y_float)
    sum_xy = np.sum(x_float * y_float)
    
    num = n * sum_xy - sum_x * sum_y
    den_x_sq = n * sum_xx - sum_x * sum_x
    den_y_sq = n * sum_yy - sum_y * sum_y
    
    # Check for zero variance. If var is zero, corr is undefined (or 0 by convention if one is const)
    if den_x_sq <= 1e-9 or den_y_sq <= 1e-9: # Using a small epsilon for float comparison
        return 0.0
    
    return num / (np.sqrt(den_x_sq) * np.sqrt(den_y_sq))


def compute_correlations_for_pairs(
    candidate_pairs: List[Tuple[int, int]],
    activations_file: Path,
    neuron_to_examples: Dict[int, Set[int]], # Needed to find common examples
    correlation_type: str = 'pearson',
    min_common_for_corr: int = 3 # Min common examples to compute correlation
) -> List[Dict]:
    """Compute correlations only for candidate pairs, optimized for memory."""
    logger.info(f"Computing {correlation_type} correlations for {len(candidate_pairs)} pairs...")
    correlations_results = []

    if not candidate_pairs:
        return []

    # Determine all unique neuron IDs involved in the candidate pairs
    involved_neurons = sorted(list(set(n_id for pair in candidate_pairs for n_id in pair)))
    if not involved_neurons:
        return []
        
    # Map these neuron IDs to continuous indices for array slicing
    neuron_id_to_slice_idx = {n_id: i for i, n_id in enumerate(involved_neurons)}

    with h5py.File(activations_file, 'r') as hf:
        if 'activations' not in hf:
            logger.error(f"'activations' dataset not found in {activations_file}")
            return []
        acts_dataset = hf['activations']
        num_total_examples_in_h5, num_total_neurons_in_h5 = acts_dataset.shape

        # Batch processing of pairs to manage loading example activations
        # This simplified approach loads all involved neurons' activations for common examples per pair.
        # A more complex batching could group pairs by common examples.
        
        for neuron_i, neuron_j in tqdm(candidate_pairs, desc="Computing correlations"):
            examples_i = neuron_to_examples.get(neuron_i, set())
            examples_j = neuron_to_examples.get(neuron_j, set())
            common_examples_ids = sorted(list(examples_i.intersection(examples_j)))

            if len(common_examples_ids) < min_common_for_corr:
                continue

            # Ensure example IDs are within bounds of the HDF5 dataset
            valid_common_examples_ids = [ex_id for ex_id in common_examples_ids if ex_id < num_total_examples_in_h5]
            if len(valid_common_examples_ids) < min_common_for_corr:
                continue
            
            # Load activations for these specific examples and the two neurons
            # This is done per pair, which might be less efficient than batching example loading,
            # but simpler and safer given HDF5's row-major storage.
            try:
                # Check if neuron_i and neuron_j are within H5 bounds
                if neuron_i >= num_total_neurons_in_h5 or neuron_j >= num_total_neurons_in_h5:
                    logger.warning(f"Neuron index out of bounds for H5 ({neuron_i} or {neuron_j} vs {num_total_neurons_in_h5}). Skipping pair.")
                    continue

                acts_i_values = acts_dataset[valid_common_examples_ids, neuron_i]
                acts_j_values = acts_dataset[valid_common_examples_ids, neuron_j]
            except IndexError as e:
                logger.warning(f"IndexError loading activations for pair ({neuron_i}, {neuron_j}) with examples {valid_common_examples_ids}: {e}")
                continue # Skip this pair

            if acts_i_values.shape[0] < min_common_for_corr: # Should be redundant due to earlier check
                continue

            try:
                corr_val = 0.0
                if correlation_type == 'pearson':
                    corr_val = fast_pearson_correlation(acts_i_values, acts_j_values)
                elif correlation_type == 'spearman':
                    # spearmanr can fail with "Too few distinct values" if inputs are constant-like
                    if len(np.unique(acts_i_values)) < 2 or len(np.unique(acts_j_values)) < 2 :
                        corr_val = 0.0 # Or handle as NaN/skip
                    else:
                        corr_val, _ = spearmanr(acts_i_values, acts_j_values)
                else:
                    logger.warning(f"Unknown correlation type: {correlation_type}")
                    continue
                
                if np.isfinite(corr_val): # Check for NaN or Inf
                    correlations_results.append({
                        'neuron_i': int(neuron_i),
                        'neuron_j': int(neuron_j),
                        'correlation': float(corr_val),
                        'n_common': len(valid_common_examples_ids)
                    })
            except Exception as e:
                logger.error(f"Error computing correlation for pair ({neuron_i}, {neuron_j}): {e}", exc_info=False) # Keep log clean
    
    return correlations_results


def compute_correlations_efficient_command(args: argparse.Namespace):
    """Main command function for computing correlations efficiently."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Executing compute_correlations_efficient_command")

    start_time_total = time.time()
    
    activations_file_path = Path(args.activations)
    neuron_data_dir_path = Path(args.neuron_data_dir)
    output_file_path = Path(args.output)

    output_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Determine number of neurons to process
    try:
        with h5py.File(activations_file_path, 'r') as hf:
            if 'activations' not in hf:
                logger.error(f"'activations' dataset not found in {activations_file_path}")
                return
            _, total_neurons_in_file = hf['activations'].shape
    except Exception as e:
        logger.error(f"Could not read HDF5 file {activations_file_path} to determine neuron count: {e}")
        return

    n_neurons_to_process = total_neurons_in_file
    if args.n_neurons is not None: # User-specified limit
        if args.n_neurons <= 0:
            logger.error(f"n_neurons must be positive, got {args.n_neurons}")
            return
        n_neurons_to_process = min(total_neurons_in_file, args.n_neurons)
    
    logger.info(f"Processing up to {n_neurons_to_process} neurons from a total of {total_neurons_in_file} in the HDF5 file.")
    
    # Step 1: Build inverted index
    # neuron_to_examples: neuron_id -> set of example_ids that are "top" for this neuron
    # example_to_neurons: example_id -> set of neuron_ids for which this example is "top"
    example_to_neurons_map, neuron_to_examples_map = build_inverted_index(
        neuron_data_dir_path, n_neurons_to_process, args.top_n_per_bin
    )
    
    if not neuron_to_examples_map:
        logger.warning("Neuron to examples map is empty. No correlations can be computed. Check neuron data files.")
        # Create empty output structure
        correlation_data_final = {
            'n_neurons_processed': n_neurons_to_process,
            'correlation_type': args.correlation_type,
            'correlation_threshold_applied': args.threshold,
            'min_overlap_for_candidate': args.min_overlap,
            'correlations': []
        }
        stats_data = {
            'total_possible_pairs': n_neurons_to_process * (n_neurons_to_process - 1) // 2,
            'candidate_pairs_checked': 0,
            'significant_correlations_found': 0
        }
    else:
        # Step 2: Find candidate pairs using the maps
        candidate_neuron_pairs = find_candidate_pairs(
            neuron_to_examples_map, args.min_overlap, args.max_pairs_per_neuron, example_to_neurons_map
        )
        
        # Step 3: Compute correlations only for these candidate pairs
        computed_correlations = compute_correlations_for_pairs(
            candidate_neuron_pairs, 
            activations_file_path, 
            neuron_to_examples_map, # Pass the map of neuron to its top example IDs
            args.correlation_type,
            min_common_for_corr=args.min_common_for_correlation # New arg
        )
        
        # Filter by the specified correlation threshold
        significant_correlations_list = [
            corr_item for corr_item in computed_correlations
            if abs(corr_item['correlation']) >= args.threshold # Use >= for threshold
        ]
        
        logger.info(f"\nFound {len(significant_correlations_list)} significant correlations (|r| >= {args.threshold})")
        
        correlation_data_final = {
            'n_neurons_processed': n_neurons_to_process,
            'correlation_type': args.correlation_type,
            'correlation_threshold_applied': args.threshold,
            'min_overlap_for_candidate': args.min_overlap,
            'min_common_for_actual_correlation': args.min_common_for_correlation,
            'correlations': significant_correlations_list
        }

        # Stats for output
        num_actual_neurons_in_candidates = len(set(n for p in candidate_neuron_pairs for n in p))
        stats_data = {
            'total_neurons_in_file': total_neurons_in_file,
            'n_neurons_processed_for_indexing': n_neurons_to_process,
            'n_neurons_with_top_example_data': len(neuron_to_examples_map),
            'n_examples_in_inverted_index': len(example_to_neurons_map),
            'total_possible_pairs_among_processed': n_neurons_to_process * (n_neurons_to_process - 1) // 2,
            'candidate_pairs_identified': len(candidate_neuron_pairs),
            'n_neurons_involved_in_candidates': num_actual_neurons_in_candidates,
            'correlations_computed_for_candidates': len(computed_correlations),
            'significant_correlations_found': len(significant_correlations_list)
        }

    # Save results (JSON format)
    with open(output_file_path, 'w') as f: # Use 'w' for text, standard json.dump
        json.dump(correlation_data_final, f, indent=2)
    logger.info(f"Saved detailed correlation results to {output_file_path}")
    
    # Save sparse matrix representation (optional, but useful)
    sparse_matrix_file_path = output_file_path.with_suffix('.npz')
    if correlation_data_final['correlations']: # Only if there are correlations
        row_indices, col_indices, data_values = [], [], []
        for corr_entry in correlation_data_final['correlations']:
            n_i, n_j = corr_entry['neuron_i'], corr_entry['neuron_j']
            val = corr_entry['correlation']
            # Add both (i,j) and (j,i) for symmetry if needed by downstream tools
            row_indices.extend([n_i, n_j])
            col_indices.extend([n_j, n_i])
            data_values.extend([val, val])
        
        # Shape of the matrix should be based on total_neurons_in_file or n_neurons_to_process
        # Using total_neurons_in_file ensures consistent matrix size if processing subsets
        matrix_shape = (total_neurons_in_file, total_neurons_in_file)
        try:
            correlation_sparse_matrix = sparse.csr_matrix(
                (data_values, (row_indices, col_indices)), shape=matrix_shape
            )
            sparse.save_npz(sparse_matrix_file_path, correlation_sparse_matrix)
            logger.info(f"Saved sparse correlation matrix to {sparse_matrix_file_path}")
        except Exception as e:
            logger.error(f"Failed to save sparse matrix: {e}")
    else:
        logger.info("No significant correlations to save in sparse matrix format.")

    stats_file_path = output_file_path.with_suffix('.stats.json')
    with open(stats_file_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    logger.info(f"Saved computation stats to {stats_file_path}")

    logger.info(f"Total computation time: {time.time() - start_time_total:.2f} seconds")
    logger.info(f"Final stats: {stats_data}")


# Example CLI setup (to be integrated into a main CLI script later)
# def main_cli_example():
#     parser = argparse.ArgumentParser(description="Compute neuron correlations efficiently.")
#     parser.add_argument("--activations", type=str, required=True, help="Path to HDF5 activations file.")
#     parser.add_argument("--neuron-data-dir", type=str, required=True, help="Directory with per-neuron JSON data.")
#     parser.add_argument("--output", type=str, required=True, help="Output JSON file for correlations.")
#     parser.add_argument("--n-neurons", type=int, default=None, help="Max number of neurons to process from HDF5.")
#     parser.add_argument("--top-n-per-bin", type=int, default=50, help="Top N examples from each activation bin to consider for indexing.") # Reduced default
#     parser.add_argument("--correlation-type", choices=['pearson', 'spearman'], default='pearson', help="Type of correlation.")
#     parser.add_argument("--threshold", type=float, default=0.3, help="Absolute correlation threshold to report.")
#     parser.add_argument("--min-overlap", type=int, default=5, help="Minimum number of common top examples for a pair to be a candidate.") # Reduced default
#     parser.add_argument("--max-pairs-per-neuron", type=int, default=50, help="Max candidate pairs to consider per neuron, sorted by overlap.") # Reduced default
#     parser.add_argument("--min-common-for-correlation", type=int, default=3, help="Minimum common examples needed to actually compute correlation for a candidate pair.")

#     cli_args = parser.parse_args()
#     compute_correlations_efficient_command(cli_args)

# if __name__ == '__main__':
#     # main_cli_example() # For direct testing
#     pass
