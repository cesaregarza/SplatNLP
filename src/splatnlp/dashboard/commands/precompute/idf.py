#!/usr/bin/env python3
"""
Precompute IDF (Inverse Document Frequency) for the entire dataset.
This is expensive but only needs to be done once.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set

import numpy as np
import polars as pl
import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_idf_from_batches(
    data_dir: Path, n_batches: int, output_path: Path
) -> pl.DataFrame:
    """
    Compute IDF from all batches without loading everything into memory.

    Args:
        data_dir: Directory with Parquet/Zarr data
        n_batches: Number of batches to process
        output_path: Where to save the IDF

    Returns:
        DataFrame with IDF values
    """
    logger.info(f"Computing IDF from {n_batches} batches...")

    # Track document frequency for each token
    token_doc_freq: Dict[int, int] = {}
    total_docs = 0

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        # Load metadata for this batch
        parquet_path = data_dir / "metadata" / f"batch_{batch_idx:04d}.parquet"

        if not parquet_path.exists():
            logger.warning(f"Batch {batch_idx} not found, skipping")
            continue

        # Load batch metadata
        batch_df = pl.read_parquet(parquet_path)

        # Process each document (sample) in the batch
        for row in batch_df.iter_rows(named=True):
            total_docs += 1

            # Get unique tokens in this document
            ability_tokens = row.get("ability_tokens", [])
            if ability_tokens:
                unique_tokens = set(ability_tokens)

                # Update document frequency for each unique token
                for token in unique_tokens:
                    token_doc_freq[token] = token_doc_freq.get(token, 0) + 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Processed {batch_idx + 1} batches, {total_docs} documents, {len(token_doc_freq)} unique tokens"
            )

    logger.info(f"Total documents: {total_docs}")
    logger.info(f"Unique tokens: {len(token_doc_freq)}")

    # Compute IDF for each token
    # IDF(t) = log(N / df(t))
    # where N is total documents and df(t) is document frequency of token t
    log_n = np.log(total_docs + 1)  # Add 1 for smoothing

    idf_data = []
    for token, doc_freq in token_doc_freq.items():
        # Standard IDF formula with smoothing
        idf = log_n - np.log(doc_freq + 1)
        idf_data.append(
            {"ability_input_tokens": token, "doc_freq": doc_freq, "idf": idf}
        )

    # Create DataFrame
    idf_df = pl.DataFrame(idf_data).sort("ability_input_tokens")

    # Save to parquet for efficient loading
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idf_df.write_parquet(output_path)
    logger.info(f"Saved IDF to {output_path}")

    # Also save summary statistics
    stats = {
        "total_documents": total_docs,
        "unique_tokens": len(token_doc_freq),
        "n_batches": n_batches,
        "most_common_tokens": idf_df.sort("doc_freq", descending=True)
        .head(10)
        .to_dicts(),
        "rarest_tokens": idf_df.sort("doc_freq").head(10).to_dicts(),
    }

    stats_path = output_path.with_suffix(".json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    return idf_df


def main():
    """Main function to compute IDF for ultra model."""
    data_dir = Path("/mnt/e/activations_ultra_efficient")
    output_dir = data_dir / "precomputed"
    output_path = output_dir / "idf.parquet"

    # Check if already computed
    if output_path.exists():
        logger.info(f"IDF already exists at {output_path}")
        response = input("Recompute? (y/n): ")
        if response.lower() != "y":
            logger.info("Loading existing IDF...")
            idf_df = pl.read_parquet(output_path)
            logger.info(f"Loaded IDF with {len(idf_df)} tokens")
            return

    # Load metadata to get batch count
    meta_path = data_dir / "conversion_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
            n_batches = metadata.get("n_batches", 6)
            total_samples = metadata.get("total_samples", 0)
            logger.info(
                f"Dataset has {n_batches} batches, {total_samples} total samples"
            )
    else:
        n_batches = 6  # Default for ultra model
        logger.warning("No metadata found, assuming 6 batches")

    # Compute IDF
    idf_df = compute_idf_from_batches(data_dir, n_batches, output_path)

    # Display summary
    print("\n" + "=" * 50)
    print("IDF Computation Complete!")
    print("=" * 50)
    print(f"Total unique tokens: {len(idf_df)}")
    print(f"Output saved to: {output_path}")
    print(f"\nTop 10 most common tokens (lowest IDF):")
    for row in idf_df.sort("idf").head(10).iter_rows(named=True):
        print(
            f"  Token {row['ability_input_tokens']}: IDF={row['idf']:.4f}, DocFreq={row['doc_freq']}"
        )
    print(f"\nTop 10 rarest tokens (highest IDF):")
    for row in (
        idf_df.sort("idf", descending=True).head(10).iter_rows(named=True)
    ):
        print(
            f"  Token {row['ability_input_tokens']}: IDF={row['idf']:.4f}, DocFreq={row['doc_freq']}"
        )


if __name__ == "__main__":
    main()
