#!/usr/bin/env python3
"""
Ultra-fast histogram computation for 24,576 features.

Key insight: Read data ONCE, compute all histograms in memory.
Instead of reading 24,576 times (once per feature), we read each batch once
and update all histograms simultaneously.
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FastHistogramComputer:
    """Compute all histograms in a single pass through the data."""

    def __init__(
        self,
        n_features: int = 24576,
        n_bins: int = 100,
        value_range: Tuple[float, float] = (0.0, 10.0),
    ):
        """
        Initialize histogram computer.

        Args:
            n_features: Number of features (24,576 for Ultra)
            n_bins: Number of histogram bins
            value_range: Expected range of values for histogram
        """
        self.n_features = n_features
        self.n_bins = n_bins
        self.value_range = value_range

        # Pre-allocate histogram arrays for ALL features
        self.hist_counts = np.zeros((n_features, n_bins), dtype=np.int64)
        self.bin_edges = np.linspace(value_range[0], value_range[1], n_bins + 1)

        # Statistics accumulators
        self.sums = np.zeros(n_features, dtype=np.float64)
        self.sum_squares = np.zeros(n_features, dtype=np.float64)
        self.counts = np.zeros(n_features, dtype=np.int64)
        self.mins = np.full(n_features, np.inf, dtype=np.float32)
        self.maxs = np.full(n_features, -np.inf, dtype=np.float32)
        self.total_samples = 0

    def process_batch(self, activations: np.ndarray):
        """
        Process a batch of activations, updating all histograms.

        Args:
            activations: Shape (n_samples, n_features)
        """
        n_samples = activations.shape[0]
        self.total_samples += n_samples

        # Process each feature in parallel using vectorized operations
        for feature_idx in range(self.n_features):
            feature_data = activations[:, feature_idx]

            # Filter non-zero values (sparse data)
            non_zero_mask = feature_data > 0
            if not np.any(non_zero_mask):
                continue

            non_zero_data = feature_data[non_zero_mask]

            # Update histogram using np.histogram with pre-defined bins
            counts, _ = np.histogram(non_zero_data, bins=self.bin_edges)
            self.hist_counts[feature_idx] += counts

            # Update statistics
            self.sums[feature_idx] += np.sum(non_zero_data)
            self.sum_squares[feature_idx] += np.sum(non_zero_data**2)
            self.counts[feature_idx] += len(non_zero_data)

            # Update min/max
            self.mins[feature_idx] = min(
                self.mins[feature_idx], np.min(non_zero_data)
            )
            self.maxs[feature_idx] = max(
                self.maxs[feature_idx], np.max(non_zero_data)
            )

    def process_batch_vectorized(self, activations: np.ndarray):
        """
        Process a batch using fully vectorized operations (experimental).

        This is faster but uses more memory.
        """
        n_samples = activations.shape[0]
        self.total_samples += n_samples

        # Create a mask for non-zero values
        non_zero_mask = activations > 0

        # Process all features at once using broadcasting
        for i in range(self.n_features):
            feature_data = activations[:, i]
            mask = non_zero_mask[:, i]

            if not np.any(mask):
                continue

            non_zero = feature_data[mask]

            # Fast histogram using searchsorted (faster than np.histogram)
            indices = np.searchsorted(self.bin_edges[1:], non_zero)
            np.add.at(self.hist_counts[i], indices, 1)

            # Update stats
            self.sums[i] += np.sum(non_zero)
            self.sum_squares[i] += np.sum(non_zero**2)
            self.counts[i] += len(non_zero)
            self.mins[i] = min(self.mins[i], np.min(non_zero))
            self.maxs[i] = max(self.maxs[i], np.max(non_zero))

    def get_results(self) -> Tuple[Dict, Dict]:
        """
        Get final histogram and statistics results.

        Returns:
            Tuple of (histograms, stats) dictionaries
        """
        histograms = {}
        stats = {}

        for i in range(self.n_features):
            if self.counts[i] > 0:
                # Compute final statistics
                mean = self.sums[i] / self.counts[i]
                variance = (self.sum_squares[i] / self.counts[i]) - (mean**2)
                std = np.sqrt(
                    max(0, variance)
                )  # Avoid negative variance due to float errors

                histograms[i] = {
                    "counts": self.hist_counts[i].tolist(),
                    "lower_bounds": self.bin_edges[:-1].tolist(),
                    "upper_bounds": self.bin_edges[1:].tolist(),
                }

                stats[i] = {
                    "min": float(self.mins[i]),
                    "max": float(self.maxs[i]),
                    "mean": float(mean),
                    "std": float(std),
                    "n_non_zero": int(self.counts[i]),
                    "n_total": self.total_samples,
                    "sparsity": float(
                        (self.total_samples - self.counts[i])
                        / self.total_samples
                    ),
                }
            else:
                # Empty feature
                histograms[i] = {
                    "counts": [0] * self.n_bins,
                    "lower_bounds": self.bin_edges[:-1].tolist(),
                    "upper_bounds": self.bin_edges[1:].tolist(),
                }
                stats[i] = {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "n_non_zero": 0,
                    "n_total": self.total_samples,
                    "sparsity": 1.0,
                }

        return histograms, stats


def compute_all_histograms_single_pass(
    data_dir: Path,
    n_batches: int,
    n_features: int = 24576,
    n_bins: int = 100,
    vectorized: bool = True,
    batch_limit: int = None,
) -> Tuple[Dict, Dict]:
    """
    Compute all histograms in a single pass through the data.

    This reads each batch file ONCE and updates all histograms simultaneously.
    """
    logger.info(f"Computing {n_features} histograms in single pass")
    logger.info(f"Method: {'vectorized' if vectorized else 'standard'}")

    # First pass: determine value range (sample a few batches)
    logger.info("Determining value range...")
    sample_max = 0.0
    for batch_idx in range(min(3, n_batches)):  # Sample first 3 batches
        zarr_path = data_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
        if zarr_path.exists():
            z = zarr.open(str(zarr_path), mode="r")
            sample_max = max(sample_max, float(np.max(z[:])))

    value_range = (0.0, sample_max * 1.1)  # Add 10% margin
    logger.info(f"Value range: {value_range}")

    # Initialize computer
    computer = FastHistogramComputer(
        n_features=n_features, n_bins=n_bins, value_range=value_range
    )

    # Process all batches
    process_method = (
        computer.process_batch_vectorized
        if vectorized
        else computer.process_batch
    )

    batch_count = batch_limit if batch_limit else n_batches
    with tqdm(total=batch_count, desc="Processing batches") as pbar:
        for batch_idx in range(batch_count):
            zarr_path = data_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
            if not zarr_path.exists():
                pbar.update(1)
                continue

            try:
                # Read entire batch at once
                z = zarr.open(str(zarr_path), mode="r")
                batch_data = z[:]  # Read all data from this batch

                # Process this batch for all features
                process_method(batch_data)

                pbar.update(1)

                # Optional: clear memory periodically
                if batch_idx % 10 == 0:
                    import gc

                    gc.collect()

            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                pbar.update(1)

    # Get results
    return computer.get_results()


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-fast histogram computation using single-pass algorithm"
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
        "--n-bins",
        type=int,
        default=100,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["standard", "vectorized"],
        default="vectorized",
        help="Processing method",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - only process first 10 batches",
    )

    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Load metadata
    meta_path = data_dir / "conversion_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    n_batches = metadata.get("n_batches", 0)
    n_features = metadata.get("n_features", 24576)

    logger.info(f"Dataset: {n_batches} batches, {n_features} features")

    # Time the computation
    start_time = time.time()

    # Compute histograms
    histograms, stats = compute_all_histograms_single_pass(
        data_dir=data_dir,
        n_batches=n_batches,
        n_features=n_features,
        n_bins=args.n_bins,
        vectorized=(args.method == "vectorized"),
        batch_limit=10 if args.test else None,
    )

    elapsed = time.time() - start_time

    # Save results
    logger.info("Saving results...")

    hist_path = output_dir / "histograms.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(histograms, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats_path = output_dir / "feature_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    print("\n" + "=" * 60)
    print("HISTOGRAM COMPUTATION COMPLETE!")
    print("=" * 60)
    print(f"Features processed: {len(histograms)}")
    print(f"Time taken: {elapsed:.1f} seconds")
    print(f"Speed: {len(histograms) / elapsed:.1f} features/second")

    if args.test:
        full_estimate = (elapsed / 10) * n_batches
        print(
            f"\nEstimated time for full dataset: {full_estimate / 60:.1f} minutes"
        )

    print(f"\nOutput saved to: {output_dir}")
    print(f"  Histograms: {hist_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Statistics: {stats_path.stat().st_size / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    exit(main())
