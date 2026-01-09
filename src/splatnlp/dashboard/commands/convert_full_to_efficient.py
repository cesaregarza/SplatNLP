#!/usr/bin/env python3
"""
Convert Full model sparse activations to efficient Zarr v3 format.

This script converts the Full model's per-feature sparse .npy files
to the same efficient Zarr + Parquet format used by Ultra model,
enabling fast mechinterp analysis.

Usage:
    poetry run python -m splatnlp.dashboard.commands.convert_full_to_efficient
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zarr
from tqdm import tqdm

# Source paths
SOURCE_DIR = Path("/mnt/e/activations2/outputs")
NEURONS_DIR = SOURCE_DIR / "neuron_acts"
ANALYSIS_DF_PATH = SOURCE_DIR / "analysis_df_records.ipc"
METADATA_PATH = SOURCE_DIR / "metadata.json"

# Target path
TARGET_DIR = Path("/mnt/e/activations_full_efficient")

# Conversion settings
BATCH_SIZE = 500_000  # Samples per batch
CHUNK_SIZE = 1000  # Rows per zarr chunk
N_FEATURES = 2048  # Full model has 2048 SAE features
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3


def get_compressor():
    """Return Zarr v3 Zstandard compressor."""
    return zarr.codecs.ZstdCodec(level=COMPRESSION_LEVEL)


def create_zarr_array(
    path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunks: tuple[int, ...],
) -> zarr.core.array.Array:
    """Create a new Zarr v3 array with compression."""
    compressor = get_compressor()
    return zarr.create_array(
        store=str(path),
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressors=compressor,
        overwrite=True,
    )


def load_feature_sparse(feature_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Load sparse activation data for a single feature.

    Returns:
        (indices, activations) - the sample indices and their activation values
    """
    neuron_dir = NEURONS_DIR / f"neuron_{feature_id:04d}"
    idxs = np.load(neuron_dir / "idxs.npy")
    acts = np.load(neuron_dir / "acts.npy")
    return idxs, acts


def convert_batch(
    analysis_df: pl.DataFrame,
    batch_idx: int,
    start_idx: int,
    end_idx: int,
    activations_dir: Path,
    metadata_dir: Path,
) -> dict[str, Any]:
    """Convert a batch of samples to Zarr + Parquet format."""
    batch_size = end_idx - start_idx

    print(f"\n=== Batch {batch_idx} (samples {start_idx:,} - {end_idx:,}) ===")

    # Build dense activation matrix for this batch
    print(
        f"  Building dense activation matrix ({batch_size} x {N_FEATURES})..."
    )
    acts_matrix = np.zeros((batch_size, N_FEATURES), dtype=np.float32)

    for feature_id in tqdm(range(N_FEATURES), desc="  Loading features"):
        idxs, acts = load_feature_sparse(feature_id)

        # Filter to batch range
        mask = (idxs >= start_idx) & (idxs < end_idx)
        batch_idxs = idxs[mask] - start_idx
        batch_acts = acts[mask]

        acts_matrix[batch_idxs, feature_id] = batch_acts.astype(np.float32)

    # Count non-zeros for stats
    n_nonzero = np.count_nonzero(acts_matrix)
    sparsity = 1.0 - (n_nonzero / acts_matrix.size)

    # Save activations as Zarr v3
    print("  Writing Zarr array...")
    zarr_path = activations_dir / f"batch_{batch_idx:04d}.zarr"
    z_acts = create_zarr_array(
        zarr_path,
        shape=acts_matrix.shape,
        dtype=acts_matrix.dtype,
        chunks=(CHUNK_SIZE, N_FEATURES),
    )

    # Write in chunks to avoid memory issues
    for chunk_start in range(0, batch_size, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, batch_size)
        z_acts[chunk_start:chunk_end, :] = acts_matrix[chunk_start:chunk_end, :]

    zarr_size = sum(
        f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
    )

    # Extract metadata for this batch
    print("  Writing Parquet metadata...")
    batch_df = analysis_df.slice(start_idx, batch_size)

    # Build clean metadata DataFrame
    clean_df = pl.DataFrame(
        {
            "batch_id": [batch_idx] * batch_size,
            "sample_id": list(range(batch_size)),
            "ability_tokens": batch_df["ability_input_tokens"].to_list(),
            "weapon_id_token": batch_df["weapon_id_token"].to_list(),
            "global_index": list(range(start_idx, end_idx)),
        }
    )

    parquet_path = metadata_dir / f"batch_{batch_idx:04d}.parquet"
    clean_df.write_parquet(
        parquet_path,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
    )
    parquet_size = parquet_path.stat().st_size

    # Free memory
    del acts_matrix
    gc.collect()

    return {
        "batch_idx": batch_idx,
        "n_samples": batch_size,
        "n_nonzero": n_nonzero,
        "sparsity": sparsity,
        "zarr_size_mb": zarr_size / (1024**2),
        "parquet_size_mb": parquet_size / (1024**2),
    }


def main():
    """Main conversion function."""
    print("=" * 60)
    print("Full Model Activation Conversion")
    print("=" * 60)

    # Load metadata
    print("\nLoading source metadata...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    n_samples = metadata["num_regular"] + metadata["num_null"]
    print(f"  Total samples: {n_samples:,}")
    print(f"  Features: {N_FEATURES}")
    print(f"  Batch size: {BATCH_SIZE:,}")

    # Load analysis DataFrame
    print("\nLoading analysis DataFrame...")
    analysis_df = pl.read_ipc(ANALYSIS_DF_PATH)
    print(f"  Loaded {len(analysis_df):,} rows")

    # Verify we have the expected number of samples
    if len(analysis_df) != n_samples:
        print(
            f"  WARNING: DataFrame has {len(analysis_df)} rows, expected {n_samples}"
        )
        n_samples = len(analysis_df)

    # Create output directories
    print(f"\nCreating output directory: {TARGET_DIR}")
    activations_dir = TARGET_DIR / "activations"
    metadata_dir = TARGET_DIR / "metadata"

    for d in [activations_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Calculate number of batches
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Will create {n_batches} batches")

    # Convert each batch
    batch_stats = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, n_samples)

        stats = convert_batch(
            analysis_df,
            batch_idx,
            start_idx,
            end_idx,
            activations_dir,
            metadata_dir,
        )
        batch_stats.append(stats)

    # Write conversion metadata
    total_zarr_mb = sum(s["zarr_size_mb"] for s in batch_stats)
    total_parquet_mb = sum(s["parquet_size_mb"] for s in batch_stats)

    conversion_metadata = {
        "n_batches": n_batches,
        "total_samples": n_samples,
        "n_features": N_FEATURES,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "compression": COMPRESSION,
        "compression_level": COMPRESSION_LEVEL,
        "total_zarr_size_mb": total_zarr_mb,
        "total_parquet_size_mb": total_parquet_mb,
        "total_output_size_mb": total_zarr_mb + total_parquet_mb,
        "batch_stats": batch_stats,
    }

    metadata_path = TARGET_DIR / "conversion_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(conversion_metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"  Batches: {n_batches}")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Zarr size: {total_zarr_mb:.1f} MB")
    print(f"  Parquet size: {total_parquet_mb:.1f} MB")
    print(f"  Total output: {total_zarr_mb + total_parquet_mb:.1f} MB")
    print(f"  Output dir: {TARGET_DIR}")


if __name__ == "__main__":
    main()
