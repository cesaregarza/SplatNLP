#!/usr/bin/env python3
"""Consolidate activation data efficiently with minimal memory usage."""

import argparse
import gc
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def consolidate_to_hdf5_streaming(
    cache_dir: Path,
    output_file: Path,
    batch_size: int = 50000
):
    """Consolidate cached chunks into HDF5 file with streaming.
    
    Args:
        cache_dir: Directory containing cached chunks
        output_file: Output HDF5 file
        batch_size: Number of examples to process at once
    """
    # Find all chunk files - try both patterns
    pkl_pattern = "activations_consolidated_batch_*_chunk_*.pkl"
    npz_pattern = "chunk_*.npz"
    
    chunk_files = sorted(cache_dir.glob(pkl_pattern))
    is_npz = False
    
    if not chunk_files:
        # Try NPZ pattern
        chunk_files = sorted(cache_dir.glob(npz_pattern))
        is_npz = True
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {cache_dir} (tried {pkl_pattern} and {npz_pattern})")
    
    print(f"Found {len(chunk_files)} chunk files ({'NPZ' if is_npz else 'PKL'} format)")
    
    # Get dimensions from first chunk
    if is_npz:
        first_chunk = np.load(chunk_files[0])
        # NPZ files use 'hidden_activations' key
        n_features = first_chunk['hidden_activations'].shape[1]
    else:
        with open(chunk_files[0], 'rb') as f:
            first_chunk = pickle.load(f)
        n_features = first_chunk['activations'].shape[1]
    
    # Count total examples
    total_examples = 0
    for chunk_file in tqdm(chunk_files, desc="Counting examples"):
        if is_npz:
            chunk_data = np.load(chunk_file)
            total_examples += len(chunk_data['hidden_activations'])
        else:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            total_examples += len(chunk_data['activations'])
    
    print(f"Total examples: {total_examples}, Features: {n_features}")
    
    # Create HDF5 file with chunked storage for efficient access
    with h5py.File(output_file, 'w') as hf:
        # Create datasets with chunking and compression
        acts_dataset = hf.create_dataset(
            'activations',
            shape=(total_examples, n_features),
            dtype='float32',
            chunks=(min(10000, total_examples), min(100, n_features)),
            compression='gzip',
            compression_opts=4
        )
        
        # Process chunks and write to HDF5
        current_idx = 0
        metadata_records = []
        
        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            if is_npz:
                chunk_data = np.load(chunk_file)
                chunk_acts = chunk_data['hidden_activations']
                
                # For NPZ files, we might not have metadata, create minimal metadata
                n_examples = len(chunk_acts)
                if 'metadata' in chunk_data:
                    # If metadata exists in NPZ
                    chunk_metadata = pd.DataFrame(chunk_data['metadata'])
                else:
                    # Create minimal metadata
                    chunk_metadata = pd.DataFrame({
                        'example_id': range(current_idx, current_idx + n_examples),
                        'chunk_file': str(chunk_file.name),
                        'chunk_idx': int(chunk_data['chunk_idx'])
                    })
                
                # Write activations
                acts_dataset[current_idx:current_idx + n_examples] = chunk_acts
                
                # Collect metadata
                metadata_records.extend(chunk_metadata.to_dict('records'))
                
            else:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                chunk_acts = chunk_data['activations']
                chunk_metadata = chunk_data['metadata']
                
                # Write activations
                n_examples = len(chunk_acts)
                acts_dataset[current_idx:current_idx + n_examples] = chunk_acts
                
                # Collect metadata
                metadata_records.extend(chunk_metadata.to_dict('records'))
            
            current_idx += n_examples
            
            # Clear memory
            if is_npz:
                del chunk_data
                del chunk_acts
                del chunk_metadata
            else:
                del chunk_data
                del chunk_acts
                del chunk_metadata
            gc.collect()
    
    # Save metadata separately
    metadata_file = output_file.with_suffix('.metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved activations to {output_file}")
    print(f"Saved metadata to {metadata_file}")
    
    return output_file, metadata_file


def extract_neuron_summaries(
    activations_file: Path,
    metadata_file: Path,
    output_dir: Path,
    n_bins: int = 10,
    top_k_per_bin: int = 1000,
    max_neurons: Optional[int] = None
):
    """Extract top examples and statistics for each neuron.
    
    Args:
        activations_file: HDF5 file with activations
        metadata_file: Pickle file with metadata
        output_dir: Output directory for neuron summaries
        n_bins: Number of activation ranges
        top_k_per_bin: Top examples per bin
        max_neurons: Maximum neurons to process
    """
    from .extract_top_examples import (
        process_activations_chunked
    )
    
    process_activations_chunked(
        activations_file,
        metadata_file,
        output_dir,
        n_bins,
        top_k_per_bin,
        chunk_size=10000,
        max_neurons=max_neurons
    )


def create_dashboard_cache(
    activations_file: Path,
    metadata_file: Path,
    neuron_data_dir: Path,
    cache_dir: Path,
    n_neurons_sample: int = 100
):
    """Create optimized cache files for dashboard.
    
    Args:
        activations_file: HDF5 file with activations
        metadata_file: Pickle file with metadata
        neuron_data_dir: Directory with neuron summaries
        cache_dir: Output directory for cache files
        n_neurons_sample: Number of neurons to include in sample
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata_records = pickle.load(f)
    
    # Create sample for quick loading
    with h5py.File(activations_file, 'r') as hf:
        acts_dataset = hf['activations']
        n_examples, n_features = acts_dataset.shape
        
        # Sample neurons evenly
        neuron_indices = np.linspace(0, n_features - 1, n_neurons_sample, dtype=int)
        
        # Sample examples
        n_examples_sample = min(50000, n_examples)
        example_indices = np.random.choice(n_examples, n_examples_sample, replace=False)
        example_indices.sort()
        
        # Create sample activation matrix
        sample_acts = np.zeros((n_examples_sample, n_neurons_sample), dtype=np.float32)
        
        for i, neuron_idx in enumerate(tqdm(neuron_indices, desc="Sampling neurons")):
            sample_acts[:, i] = acts_dataset[example_indices, neuron_idx]
    
    # Save sample data
    sample_file = cache_dir / 'dashboard_sample.npz'
    np.savez_compressed(
        sample_file,
        activations=sample_acts,
        neuron_indices=neuron_indices,
        example_indices=example_indices
    )
    
    # Save sample metadata
    sample_metadata = [metadata_records[i] for i in example_indices]
    sample_metadata_file = cache_dir / 'dashboard_sample_metadata.pkl'
    with open(sample_metadata_file, 'wb') as f:
        pickle.dump(sample_metadata, f)
    
    # Create index file for quick lookups
    index_data = {
        'n_neurons': int(n_features),
        'n_examples': int(n_examples),
        'n_neurons_sample': int(n_neurons_sample),
        'n_examples_sample': int(n_examples_sample),
        'neuron_data_dir': str(neuron_data_dir),
        'activations_file': str(activations_file),
        'metadata_file': str(metadata_file)
    }
    
    index_file = cache_dir / 'dashboard_index.json'
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"Created dashboard cache in {cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate activations efficiently"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Consolidate command
    consolidate_parser = subparsers.add_parser(
        'consolidate',
        help='Consolidate chunks to HDF5'
    )
    consolidate_parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Directory with cached chunks'
    )
    consolidate_parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output HDF5 file'
    )
    consolidate_parser.add_argument(
        '--batch-size',
        type=int,
        default=50000,
        help='Batch size for processing'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract neuron summaries'
    )
    extract_parser.add_argument(
        '--activations',
        type=Path,
        required=True,
        help='HDF5 file with activations'
    )
    extract_parser.add_argument(
        '--metadata',
        type=Path,
        required=True,
        help='Pickle file with metadata'
    )
    extract_parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for summaries'
    )
    extract_parser.add_argument(
        '--n-bins',
        type=int,
        default=10,
        help='Number of activation ranges'
    )
    extract_parser.add_argument(
        '--top-k',
        type=int,
        default=1000,
        help='Top examples per bin'
    )
    extract_parser.add_argument(
        '--max-neurons',
        type=int,
        default=None,
        help='Maximum neurons to process'
    )
    
    # Cache command
    cache_parser = subparsers.add_parser(
        'cache',
        help='Create dashboard cache'
    )
    cache_parser.add_argument(
        '--activations',
        type=Path,
        required=True,
        help='HDF5 file with activations'
    )
    cache_parser.add_argument(
        '--metadata',
        type=Path,
        required=True,
        help='Pickle file with metadata'
    )
    cache_parser.add_argument(
        '--neuron-data-dir',
        type=Path,
        required=True,
        help='Directory with neuron summaries'
    )
    cache_parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Output cache directory'
    )
    cache_parser.add_argument(
        '--n-neurons-sample',
        type=int,
        default=100,
        help='Number of neurons in sample'
    )
    
    args = parser.parse_args()
    
    if args.command == 'consolidate':
        consolidate_to_hdf5_streaming(
            args.cache_dir,
            args.output,
            args.batch_size
        )
    elif args.command == 'extract':
        from .extract_top_examples import process_activations_chunked
        process_activations_chunked(
            args.activations,
            args.metadata,
            args.output_dir,
            args.n_bins,
            args.top_k,
            chunk_size=10000,
            max_neurons=args.max_neurons
        )
    elif args.command == 'cache':
        create_dashboard_cache(
            args.activations,
            args.metadata,
            args.neuron_data_dir,
            args.cache_dir,
            args.n_neurons_sample
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()