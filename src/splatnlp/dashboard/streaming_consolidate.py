#!/usr/bin/env python3
"""
Streaming consolidation that processes data without loading into memory.

This script processes activation chunks one row at a time, inserting directly
into the database without building large matrices in memory.
"""

import argparse
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from splatnlp.dashboard.database_manager import DashboardDatabase

logger = logging.getLogger(__name__)


def stream_chunk_data(chunk_file: Path) -> Iterator[Tuple[int, Dict, np.ndarray]]:
    """Stream data from a chunk file one example at a time.
    
    Args:
        chunk_file: Path to chunk file
        
    Yields:
        Tuple of (example_index, metadata_dict, activation_vector)
    """
    if chunk_file.suffix == '.npz':
        # Handle NPZ files
        with np.load(chunk_file) as data:
            activations = data['hidden_activations']
            n_examples = activations.shape[0]
            
            for i in range(n_examples):
                metadata = {
                    'weapon_id': -1,
                    'weapon_name': 'Unknown',
                    'input_abilities_str': '',
                    'top_predicted_abilities_str': '',
                    'ability_input_tokens': [],
                    'is_null_token': False,
                    'chunk_file': chunk_file.name
                }
                yield i, metadata, activations[i]
                
    else:
        # Handle PKL files
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
        
        activations = chunk_data['activations']
        metadata_df = chunk_data['metadata']
        
        for i in range(len(activations)):
            metadata = metadata_df.iloc[i].to_dict()
            metadata['chunk_file'] = chunk_file.name
            yield i, metadata, activations[i]


def process_single_chunk_streaming(
    chunk_file: Path,
    db: DashboardDatabase,
    starting_example_id: int,
    activation_threshold: float = 1e-6,
    batch_size: int = 1000
) -> int:
    """Process a single chunk file in streaming fashion.
    
    Args:
        chunk_file: Path to chunk file
        db: Database manager
        starting_example_id: Starting example ID for this chunk
        activation_threshold: Minimum activation to store
        batch_size: Number of examples to batch before database insert
        
    Returns:
        Number of examples processed
    """
    examples_batch = []
    activations_batch = []
    example_ids_batch = []
    current_example_id = starting_example_id
    examples_processed = 0
    
    logger.info(f"Streaming chunk {chunk_file.name}")
    
    for chunk_idx, metadata, activation_vector in stream_chunk_data(chunk_file):
        # Prepare example data
        example_data = {
            'id': current_example_id,
            'weapon_id': metadata.get('weapon_id', -1),
            'weapon_name': metadata.get('weapon_name', 'Unknown'),
            'ability_input_tokens': metadata.get('ability_input_tokens', []),
            'input_abilities_str': metadata.get('input_abilities_str', ''),
            'top_predicted_abilities_str': metadata.get('top_predicted_abilities_str', ''),
            'is_null_token': metadata.get('is_null_token', False),
            'metadata': {k: v for k, v in metadata.items() if k not in [
                'weapon_id', 'weapon_name', 'ability_input_tokens', 
                'input_abilities_str', 'top_predicted_abilities_str', 'is_null_token'
            ]}
        }
        
        examples_batch.append(example_data)
        activations_batch.append(activation_vector)
        example_ids_batch.append(current_example_id)
        
        current_example_id += 1
        examples_processed += 1
        
        # Process batch when it reaches batch_size
        if len(examples_batch) >= batch_size:
            # Insert examples
            db.insert_examples_batch(examples_batch)
            
            # Insert activations (convert list to numpy array for efficient processing)
            activations_array = np.stack(activations_batch)
            db.insert_activations_batch(
                activations_array, 
                example_ids_batch, 
                activation_threshold=activation_threshold
            )
            
            # Clear batches
            examples_batch.clear()
            activations_batch.clear()
            example_ids_batch.clear()
            
            # Force garbage collection
            gc.collect()
    
    # Process remaining batch
    if examples_batch:
        db.insert_examples_batch(examples_batch)
        activations_array = np.stack(activations_batch)
        db.insert_activations_batch(
            activations_array, 
            example_ids_batch, 
            activation_threshold=activation_threshold
        )
    
    logger.info(f"Processed {examples_processed} examples from {chunk_file.name}")
    return examples_processed


def streaming_consolidate(
    cache_dir: Path,
    db_path: Path,
    activation_threshold: float = 1e-6,
    batch_size: int = 1000,
    resume: bool = True
) -> None:
    """Consolidate chunks to database using streaming approach.
    
    Args:
        cache_dir: Directory containing cached chunks
        db_path: Path to output database
        activation_threshold: Minimum activation value to store
        batch_size: Examples to batch before database insert
    """
    # Initialize database
    db = DashboardDatabase(db_path)
    
    # Find chunk files
    chunk_files = list(cache_dir.glob("*.pkl")) + list(cache_dir.glob("*.npz"))
    chunk_files = sorted(chunk_files)
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {cache_dir}")
    
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    # Check for existing data if resume is enabled
    processed_chunks = set()
    current_example_id = 0
    
    if resume and db_path.exists():
        with db.get_connection() as conn:
            # Get max example ID to determine where to resume
            cursor = conn.execute("SELECT MAX(id) FROM examples")
            max_id = cursor.fetchone()[0]
            if max_id is not None:
                current_example_id = max_id + 1
                logger.info(f"Resuming from example ID {current_example_id}")
                
                # Get list of processed chunk files from metadata
                cursor = conn.execute("""
                    SELECT DISTINCT json_extract(metadata, '$.chunk_file') as chunk_file 
                    FROM examples 
                    WHERE json_extract(metadata, '$.chunk_file') IS NOT NULL
                """)
                for row in cursor.fetchall():
                    if row[0]:
                        processed_chunks.add(row[0])
                
                logger.info(f"Found {len(processed_chunks)} already processed chunks")
    
    # Filter out already processed chunks
    if processed_chunks:
        remaining_chunks = [f for f in chunk_files if f.name not in processed_chunks]
        logger.info(f"Skipping {len(processed_chunks)} processed chunks, processing {len(remaining_chunks)} remaining")
        chunk_files = remaining_chunks
    
    total_examples = 0
    
    # Process each chunk in streaming fashion
    for chunk_file in tqdm(chunk_files, desc="Streaming chunks to database"):
        try:
            examples_processed = process_single_chunk_streaming(
                chunk_file,
                db,
                current_example_id,
                activation_threshold,
                batch_size
            )
            current_example_id += examples_processed
            total_examples += examples_processed
            
        except Exception as e:
            logger.error(f"Error processing {chunk_file}: {e}")
            continue
    
    logger.info(f"Streaming consolidation complete. Processed {total_examples} examples")
    
    # Get database info
    info = db.get_database_info()
    logger.info(f"Database info: {info}")


def compute_analytics_streaming(
    db_path: Path,
    features_per_batch: int = 100,
    max_features: Optional[int] = None
) -> None:
    """Compute analytics in streaming fashion without loading all data.
    
    Args:
        db_path: Path to database
        features_per_batch: Features to process per batch
        max_features: Maximum features to process (for testing)
    """
    db = DashboardDatabase(db_path)
    
    # Get feature IDs in batches
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT feature_id FROM activations ORDER BY feature_id")
        all_feature_ids = [row[0] for row in cursor.fetchall()]
    
    if max_features:
        all_feature_ids = all_feature_ids[:max_features]
    
    logger.info(f"Computing analytics for {len(all_feature_ids)} features in batches of {features_per_batch}")
    
    # Process features in batches
    for i in tqdm(range(0, len(all_feature_ids), features_per_batch), desc="Processing feature batches"):
        batch_features = all_feature_ids[i:i + features_per_batch]
        
        for feature_id in batch_features:
            try:
                # Compute statistics (this queries only one feature at a time)
                db.compute_and_store_feature_stats(feature_id)
                
                # Get activations for top examples (only loads one feature)
                activations, example_ids = db.get_feature_activations(feature_id)
                
                if len(activations) > 0:
                    # Store top examples
                    db.store_top_examples(
                        feature_id,
                        example_ids,
                        activations.tolist(),
                        top_k=50
                    )
                
            except Exception as e:
                logger.error(f"Error processing feature {feature_id}: {e}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    logger.info("Analytics computation complete")


def main():
    """Main streaming consolidation script."""
    parser = argparse.ArgumentParser(
        description="Stream chunks to database without memory loading"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Streaming consolidate command
    consolidate_parser = subparsers.add_parser(
        'stream',
        help='Stream chunks to database'
    )
    consolidate_parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Directory with cached chunks'
    )
    consolidate_parser.add_argument(
        '--output-db',
        type=Path,
        required=True,
        help='Output database path'
    )
    consolidate_parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Examples to batch before database insert'
    )
    consolidate_parser.add_argument(
        '--activation-threshold',
        type=float,
        default=1e-6,
        help='Minimum activation value to store'
    )
    
    # Streaming analytics command
    analytics_parser = subparsers.add_parser(
        'analytics',
        help='Compute analytics in streaming fashion'
    )
    analytics_parser.add_argument(
        '--db-path',
        type=Path,
        required=True,
        help='Database path'
    )
    analytics_parser.add_argument(
        '--features-per-batch',
        type=int,
        default=100,
        help='Features to process per batch'
    )
    analytics_parser.add_argument(
        '--max-features',
        type=int,
        default=None,
        help='Maximum features to process'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'stream':
        streaming_consolidate(
            args.cache_dir,
            args.output_db,
            args.activation_threshold,
            args.batch_size
        )
    elif args.command == 'analytics':
        compute_analytics_streaming(
            args.db_path,
            args.features_per_batch,
            args.max_features
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()