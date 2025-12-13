#!/usr/bin/env python3
"""
Transpose activation data for ultra-fast feature-wise access.

The problem: Zarr stores are row‑major (samples × features), but we need
column‑major access (features × samples) for histogram computation.

Solution: Create a transposed copy of the data once, then histogram
computation becomes trivial and ultra‑fast.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import (  # v3 compressors live in zarr.codecs
    BloscCodec,
    BloscShuffle,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transpose_activations(
    source_dir: Path,
    target_dir: Path,
    n_features: int = 24576,
    chunk_features: int = 100,
):
    """
    Create a transposed copy of activation data.

    Instead of (samples, features), we create (features, samples).
    This allows direct feature‑wise access.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = source_dir / "conversion_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    n_batches = metadata.get("n_batches", 0)

    # First, determine total samples
    logger.info("Counting total samples...")
    total_samples = 0
    batch_sizes = []
    for batch_idx in range(n_batches):
        zarr_path = source_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
        if zarr_path.exists():
            z = zarr.open_array(str(zarr_path), mode="r")
            batch_size = z.shape[0]
            batch_sizes.append(batch_size)
            total_samples += batch_size

    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total features: {n_features}")

    # Create transposed zarr array (features, samples).
    # Zarr 3 removed DirectoryStore in favour of LocalStore【61903600004094†L214-L233】.
    # We use a LocalStore for persistence and specify the compressor via the
    # `compressors` keyword. With Zarr‑Python 3, compressors come from
    # `zarr.codecs` rather than `numcodecs` and are passed via the `compressors`
    # argument【698954283501210†L218-L237】. Using LocalStore makes this code forward
    # compatible with Zarr v3 while retaining the v2 directory semantics.
    store = zarr.storage.LocalStore(
        str(target_dir / "transposed_activations.zarr")
    )
    # Use the compressors keyword introduced in Zarr‑Python 3. The BloscCodec lives
    # in zarr.codecs; we set the clevel and cname similar to the old numcodecs
    # compressor. If you need bit/byte shuffling, you can set the shuffle argument
    # accordingly (e.g. BloscShuffle.bitshuffle)【698954283501210†L218-L237】.
    blosc_compressor = BloscCodec(
        cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle
    )
    transposed = zarr.create_array(
        store=store,
        shape=(n_features, total_samples),
        chunks=(1, 10000),  # Each chunk is 1 feature × 10k samples
        dtype="float32",
        compressors=blosc_compressor,
        overwrite=True,
    )

    # Process in feature chunks to manage memory
    for feature_start in tqdm(
        range(0, n_features, chunk_features), desc="Transposing features"
    ):
        feature_end = min(feature_start + chunk_features, n_features)
        feature_range = range(feature_start, feature_end)

        # Collect all data for this feature chunk
        chunk_data = np.zeros(
            (len(feature_range), total_samples), dtype=np.float32
        )

        sample_offset = 0
        for batch_idx, batch_size in enumerate(batch_sizes):
            zarr_path = (
                source_dir / "activations" / f"batch_{batch_idx:04d}.zarr"
            )
            if zarr_path.exists():
                z = zarr.open_array(str(zarr_path), mode="r")
                # Read this feature chunk from the batch
                batch_data = z[:, feature_start:feature_end]
                # Transpose and store
                chunk_data[:, sample_offset : sample_offset + batch_size] = (
                    batch_data.T
                )
                sample_offset += batch_size

        # Write to transposed array
        transposed[feature_start:feature_end, :] = chunk_data

    # Save metadata
    trans_metadata = {
        "n_features": n_features,
        "n_samples": total_samples,
        "source_batches": n_batches,
        "format": "transposed",
        "shape": [n_features, total_samples],
        "description": "Transposed activations for fast feature-wise access",
    }

    with open(target_dir / "transposed_metadata.json", "w") as f:
        json.dump(trans_metadata, f, indent=2)

    logger.info(f"Transposed data saved to {target_dir}")
    return transposed


def compute_histograms_from_transposed(
    transposed_path: Path, n_features: int = 24576, n_bins: int = 100
):
    """
    Compute histograms from transposed data - ULTRA FAST!

    Each feature is now a contiguous array, so we can compute
    histograms with direct access.
    """
    import pickle

    # Open transposed array using the convenience function. Using open_array will
    # read existing metadata automatically.
    z = zarr.open_array(
        str(transposed_path / "transposed_activations.zarr"), mode="r"
    )

    histograms = {}
    stats = {}

    logger.info(f"Computing {n_features} histograms from transposed data...")

    for feature_idx in tqdm(range(n_features), desc="Computing histograms"):
        # Direct access to all samples for this feature!
        feature_data = z[feature_idx, :]

        # Filter non-zero (sparse data)
        non_zero = feature_data[feature_data > 0]

        if len(non_zero) > 0:
            # Compute histogram
            counts, bins = np.histogram(non_zero, bins=n_bins)

            histograms[feature_idx] = {
                "counts": counts.tolist(),
                "lower_bounds": bins[:-1].tolist(),
                "upper_bounds": bins[1:].tolist(),
            }

            stats[feature_idx] = {
                "min": float(non_zero.min()),
                "max": float(non_zero.max()),
                "mean": float(non_zero.mean()),
                "std": float(non_zero.std()),
                "n_non_zero": len(non_zero),
                "n_total": len(feature_data),
                "sparsity": float(
                    (len(feature_data) - len(non_zero)) / len(feature_data)
                ),
            }
        else:
            histograms[feature_idx] = {
                "counts": [0] * n_bins,
                "lower_bounds": [0.0] * n_bins,
                "upper_bounds": [0.0] * n_bins,
            }
            stats[feature_idx] = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "n_non_zero": 0,
                "n_total": len(feature_data),
                "sparsity": 1.0,
            }

    # Save
    output_dir = transposed_path.parent / "precomputed"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "histograms.pkl", "wb") as f:
        pickle.dump(histograms, f)

    with open(output_dir / "feature_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Saved to {output_dir}")
    return histograms, stats


def main():
    parser = argparse.ArgumentParser(
        description="Transpose activations for ultra-fast histogram computation"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/e/activations_ultra_efficient",
        help="Source directory with Zarr files",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/mnt/e/activations_ultra_transposed",
        help="Target directory for transposed data",
    )
    parser.add_argument(
        "--compute-histograms",
        action="store_true",
        help="Compute histograms after transposing",
    )
    parser.add_argument(
        "--skip-transpose",
        action="store_true",
        help="Skip transposition if already done",
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    if not args.skip_transpose:
        # Step 1: Transpose the data (one-time cost)
        logger.info("Step 1: Transposing activation data...")
        start = time.time()
        transpose_activations(source_dir, target_dir)
        logger.info(f"Transposition took {time.time() - start:.1f} seconds")

    if args.compute_histograms or args.skip_transpose:
        # Step 2: Compute histograms (now ultra-fast!)
        logger.info("Step 2: Computing histograms from transposed data...")
        start = time.time()
        compute_histograms_from_transposed(target_dir)
        elapsed = time.time() - start

        logger.info(f"Histogram computation took {elapsed:.1f} seconds")
        logger.info(f"Speed: {24576 / elapsed:.1f} features/second")

    return 0


if __name__ == "__main__":
    exit(main())
