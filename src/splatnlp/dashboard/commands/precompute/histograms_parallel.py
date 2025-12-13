#!/usr/bin/env python3
"""
Fast parallel histogram computation for Ultra model (24,576 features).

Uses multiprocessing and batch processing to dramatically speed up computation.
"""

import argparse
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_feature_batch(
    data_dir: Path,
    feature_ids: List[int],
    n_batches: int,
    nb_bins: int = 100,
    batch_cache: Optional[Dict] = None,
) -> Dict:
    """
    Process a batch of features in a single worker.

    This function is designed to be called by multiprocessing workers.
    """
    results = {}

    # Pre-load all batch files once for this worker
    if batch_cache is None:
        batch_cache = {}
        for batch_idx in range(n_batches):
            zarr_path = data_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
            if zarr_path.exists():
                try:
                    batch_cache[batch_idx] = zarr.open(str(zarr_path), mode="r")
                except Exception as e:
                    logger.warning(f"Failed to load batch {batch_idx}: {e}")

    # Process each feature in this batch
    for feature_id in feature_ids:
        all_acts = []
        total_samples = 0

        # Collect activations from all batches
        for batch_idx, z in batch_cache.items():
            try:
                feature_acts = z[:, feature_id]
                total_samples += len(feature_acts)
                non_zero_acts = feature_acts[feature_acts > 0]
                if len(non_zero_acts) > 0:
                    all_acts.append(non_zero_acts)
            except Exception as e:
                logger.debug(
                    f"Error reading feature {feature_id} from batch {batch_idx}: {e}"
                )

        # Compute histogram if we have data
        if all_acts:
            acts = np.concatenate(all_acts).astype(np.float32)
            counts, bins = np.histogram(acts, bins=nb_bins)

            results[feature_id] = {
                "counts": counts.tolist(),
                "lower_bounds": bins[:-1].tolist(),
                "upper_bounds": bins[1:].tolist(),
                "min": float(acts.min()),
                "max": float(acts.max()),
                "mean": float(acts.mean()),
                "std": float(acts.std()),
                "n_non_zero": len(acts),
                "n_total": total_samples,
                "sparsity": (
                    float((total_samples - len(acts)) / total_samples)
                    if total_samples > 0
                    else 1.0
                ),
            }
        else:
            # Feature has no activations
            results[feature_id] = {
                "counts": [0] * nb_bins,
                "lower_bounds": [0.0] * nb_bins,
                "upper_bounds": [0.0] * nb_bins,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "n_non_zero": 0,
                "n_total": total_samples,
                "sparsity": 1.0,
            }

    return results


def compute_histograms_parallel(
    data_dir: Path,
    n_features: int,
    n_batches: int,
    nb_bins: int = 100,
    n_workers: int = None,
    chunk_size: int = 100,
) -> Tuple[Dict, Dict]:
    """
    Compute histograms in parallel using multiprocessing.

    Args:
        data_dir: Directory containing Zarr files
        n_features: Total number of features
        n_batches: Number of batch files
        nb_bins: Number of histogram bins
        n_workers: Number of parallel workers (None = CPU count)
        chunk_size: Number of features per worker chunk

    Returns:
        Tuple of (histograms, stats) dictionaries
    """
    import multiprocessing

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 16)  # Cap at 16 workers

    logger.info(f"Using {n_workers} parallel workers, chunk size {chunk_size}")

    # Create feature chunks for parallel processing
    feature_chunks = []
    for i in range(0, n_features, chunk_size):
        chunk = list(range(i, min(i + chunk_size, n_features)))
        feature_chunks.append(chunk)

    logger.info(
        f"Created {len(feature_chunks)} chunks for {n_features} features"
    )

    # Process chunks in parallel
    histograms = {}
    stats = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_feature_batch,
                data_dir,
                chunk,
                n_batches,
                nb_bins,
                None,  # Let each worker load its own cache
            ): chunk_idx
            for chunk_idx, chunk in enumerate(feature_chunks)
        }

        # Process results as they complete
        with tqdm(total=len(feature_chunks), desc="Processing chunks") as pbar:
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    chunk_results = future.result()

                    # Merge results
                    for feature_id, data in chunk_results.items():
                        # Separate histogram and stats
                        hist_data = {
                            k: v
                            for k, v in data.items()
                            if k in ["counts", "lower_bounds", "upper_bounds"]
                        }
                        stat_data = {
                            k: v
                            for k, v in data.items()
                            if k
                            not in ["counts", "lower_bounds", "upper_bounds"]
                        }

                        histograms[feature_id] = hist_data
                        stats[feature_id] = stat_data

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                    pbar.update(1)

    return histograms, stats


def compute_histograms_vectorized(
    data_dir: Path,
    n_features: int,
    n_batches: int,
    nb_bins: int = 100,
    batch_size: int = 100,
) -> Tuple[Dict, Dict]:
    """
    Alternative: Compute histograms using vectorized operations.

    Process multiple features at once within each batch file.
    """
    logger.info(f"Using vectorized computation, batch size {batch_size}")

    histograms = {}
    stats = {}

    # Process features in batches
    for feature_start in tqdm(
        range(0, n_features, batch_size), desc="Feature batches"
    ):
        feature_end = min(feature_start + batch_size, n_features)
        feature_range = range(feature_start, feature_end)

        # Accumulate data for this feature batch
        feature_data = {fid: [] for fid in feature_range}
        feature_totals = {fid: 0 for fid in feature_range}

        # Read all batches once for this feature range
        for batch_idx in range(n_batches):
            zarr_path = data_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
            if zarr_path.exists():
                try:
                    z = zarr.open(str(zarr_path), mode="r")

                    # Read all features in range at once
                    batch_data = z[:, feature_start:feature_end]

                    for i, fid in enumerate(feature_range):
                        feature_acts = batch_data[:, i]
                        feature_totals[fid] += len(feature_acts)

                        non_zero = feature_acts[feature_acts > 0]
                        if len(non_zero) > 0:
                            feature_data[fid].append(non_zero)

                except Exception as e:
                    logger.debug(f"Error reading batch {batch_idx}: {e}")

        # Compute histograms for this batch of features
        for fid in feature_range:
            if feature_data[fid]:
                acts = np.concatenate(feature_data[fid]).astype(np.float32)
                counts, bins = np.histogram(acts, bins=nb_bins)

                histograms[fid] = {
                    "counts": counts.tolist(),
                    "lower_bounds": bins[:-1].tolist(),
                    "upper_bounds": bins[1:].tolist(),
                }

                stats[fid] = {
                    "min": float(acts.min()),
                    "max": float(acts.max()),
                    "mean": float(acts.mean()),
                    "std": float(acts.std()),
                    "n_non_zero": len(acts),
                    "n_total": feature_totals[fid],
                    "sparsity": (
                        float(
                            (feature_totals[fid] - len(acts))
                            / feature_totals[fid]
                        )
                        if feature_totals[fid] > 0
                        else 1.0
                    ),
                }
            else:
                # Empty feature
                histograms[fid] = {
                    "counts": [0] * nb_bins,
                    "lower_bounds": [0.0] * nb_bins,
                    "upper_bounds": [0.0] * nb_bins,
                }
                stats[fid] = {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "n_non_zero": 0,
                    "n_total": feature_totals[fid],
                    "sparsity": 1.0,
                }

    return histograms, stats


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel histogram computation for Ultra model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/e/activations_ultra_efficient",
        help="Path to efficient database directory with Zarr files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/precomputed_ultra",
        help="Output directory for precomputed data",
    )
    parser.add_argument(
        "--nb-hist-bins",
        type=int,
        default=100,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["parallel", "vectorized"],
        default="parallel",
        help="Computation method to use",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Features per chunk for parallel processing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - only process first 1000 features",
    )

    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    meta_path = data_dir / "conversion_metadata.json"
    if not meta_path.exists():
        logger.error(f"Metadata not found: {meta_path}")
        return 1

    # Load metadata
    with open(meta_path) as f:
        metadata = json.load(f)

    n_features = 24576  # Ultra model
    n_batches = metadata.get("n_batches", 0)

    if args.test:
        n_features = min(1000, n_features)
        logger.info("TEST MODE: Processing only first 1000 features")

    logger.info(f"Processing {n_features} features from {n_batches} batches")
    logger.info(f"Method: {args.method}")

    # Time the computation
    start_time = time.time()

    # Compute histograms
    if args.method == "parallel":
        histograms, stats = compute_histograms_parallel(
            data_dir,
            n_features,
            n_batches,
            args.nb_hist_bins,
            args.n_workers,
            args.chunk_size,
        )
    else:
        histograms, stats = compute_histograms_vectorized(
            data_dir,
            n_features,
            n_batches,
            args.nb_hist_bins,
            batch_size=args.chunk_size,
        )

    elapsed = time.time() - start_time

    # Save results
    logger.info("Saving results...")

    hist_path = output_dir / "histograms.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(histograms, f)

    stats_path = output_dir / "feature_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # Summary
    logger.info("=" * 60)
    logger.info("Computation Complete!")
    logger.info("=" * 60)
    logger.info(f"Processed: {len(histograms)} features")
    logger.info(f"Time taken: {elapsed:.1f} seconds")
    logger.info(f"Rate: {len(histograms) / elapsed:.1f} features/second")
    logger.info(
        f"Estimated time for 24,576 features: {24576 / (len(histograms) / elapsed) / 60:.1f} minutes"
    )
    logger.info(f"Output saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
