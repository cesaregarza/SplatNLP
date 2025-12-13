#!/usr/bin/env python3
"""
Optimized example storage that processes all features in a single pass.
Much faster than per-feature approach.
"""

import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import orjson
import polars as pl
import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Example:
    """Single example with activation."""

    activation: float
    batch_id: int
    sample_id: int
    global_index: int
    ability_tokens: List[int]
    weapon_id: int
    is_null: bool

    def __lt__(self, other):
        # For heap comparison (min-heap by default, so we use negative)
        return self.activation < other.activation


class OptimizedExampleStorage:
    """Build example storage in a single pass through the data."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        top_k: int = 100,
        activation_threshold: float = 0.1,
        n_features: int = 24576,
        include_negative: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.threshold = activation_threshold
        self.n_features = n_features
        self.include_negative = include_negative

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        with open(self.data_dir / "conversion_metadata.json") as f:
            self.meta = orjson.loads(f.read())

        # Initialize top-k heaps for each feature
        # Using min-heap to efficiently maintain top-k largest
        self.feature_heaps = [[] for _ in range(n_features)]
        self.feature_stats = [
            {"count": 0, "sum": 0.0} for _ in range(n_features)
        ]

    def process_all_batches(self):
        """Process all batches in a single pass."""

        logger.info(f"Processing {self.meta['n_batches']} batches...")

        for batch_idx in tqdm(
            range(self.meta["n_batches"]), desc="Processing batches"
        ):
            self._process_batch(batch_idx)

        logger.info("All batches processed, saving results...")
        self._save_results()

    def _process_batch(self, batch_idx: int):
        """Process a single batch, updating all feature heaps."""

        logger.info(f"Processing batch {batch_idx}...")

        # Load metadata
        meta_df = pl.read_parquet(
            self.data_dir / "metadata" / f"batch_{batch_idx:04d}.parquet"
        )

        # Load activations
        z = zarr.open(
            str(self.data_dir / "activations" / f"batch_{batch_idx:04d}.zarr"),
            mode="r",
        )

        # Process in chunks to manage memory
        chunk_size = 5000
        n_samples = z.shape[0]
        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        for chunk_idx, chunk_start in enumerate(
            range(0, n_samples, chunk_size)
        ):
            chunk_end = min(chunk_start + chunk_size, n_samples)

            # Load chunk of activations
            act_chunk = z[chunk_start:chunk_end]

            # Get metadata for this chunk
            chunk_meta = meta_df[chunk_start:chunk_end]

            # Find all non-zero activations efficiently
            # This gives us (sample_idx, feature_idx) pairs where activation > threshold
            if self.include_negative:
                # Include both positive and negative activations
                sample_indices, feature_indices = np.where(
                    np.abs(act_chunk) > self.threshold
                )
            else:
                # Only positive activations
                sample_indices, feature_indices = np.where(
                    act_chunk > self.threshold
                )

            if chunk_idx % 5 == 0:
                logger.info(
                    f"  Batch {batch_idx}, chunk {chunk_idx}/{n_chunks}: found {len(sample_indices)} activations"
                )

            # Process each activation
            for sample_idx, feature_idx in zip(sample_indices, feature_indices):
                # Skip features beyond our limit
                if feature_idx >= self.n_features:
                    continue

                activation = float(act_chunk[sample_idx, feature_idx])

                # Update statistics
                self.feature_stats[feature_idx]["count"] += 1
                self.feature_stats[feature_idx]["sum"] += activation

                # Get metadata for this sample (convert numpy int to Python int for Polars)
                row = chunk_meta[int(sample_idx)]

                # Create example (extract values from Polars row)
                # Handle the ability_tokens extraction properly
                ability_tokens_val = row["ability_tokens"].item()
                if hasattr(ability_tokens_val, "to_list"):
                    ability_tokens_val = ability_tokens_val.to_list()
                elif hasattr(ability_tokens_val, "tolist"):
                    ability_tokens_val = ability_tokens_val.tolist()

                example = Example(
                    activation=activation,
                    batch_id=batch_idx,
                    sample_id=chunk_start + sample_idx,
                    global_index=int(row["global_index"].item()),
                    ability_tokens=ability_tokens_val,
                    weapon_id=int(row["weapon_id_token"].item()),
                    is_null=bool(row["is_null_token"].item()),
                )

                # Update top-k heap for this feature
                heap = self.feature_heaps[feature_idx]

                if len(heap) < self.top_k:
                    heapq.heappush(heap, example)
                elif activation > heap[0].activation:
                    heapq.heapreplace(heap, example)

    def _save_results(self):
        """Save all results efficiently."""

        # Prepare data for consolidated storage
        all_examples = []
        feature_index = []

        logger.info("Consolidating examples...")

        for feature_id in tqdm(
            range(self.n_features), desc="Consolidating features"
        ):
            heap = self.feature_heaps[feature_id]

            if not heap:
                continue

            # Sort examples by activation (descending)
            examples = sorted(heap, key=lambda x: x.activation, reverse=True)

            # Track index info
            start_idx = len(all_examples)

            # Add examples to consolidated list
            for example in examples:
                # Ensure ability_tokens is a list
                ability_tokens = example.ability_tokens
                if hasattr(ability_tokens, "to_list"):
                    ability_tokens = ability_tokens.to_list()
                elif hasattr(ability_tokens, "tolist"):
                    ability_tokens = ability_tokens.tolist()

                all_examples.append(
                    {
                        "feature_id": feature_id,
                        "activation": example.activation,
                        "batch_id": example.batch_id,
                        "sample_id": example.sample_id,
                        "global_index": example.global_index,
                        "ability_tokens_json": orjson.dumps(
                            ability_tokens
                        ).decode("utf-8"),
                        "weapon_id": example.weapon_id,
                        "is_null": example.is_null,
                    }
                )

            # Add to index
            stats = self.feature_stats[feature_id]
            feature_index.append(
                {
                    "feature_id": feature_id,
                    "start_idx": start_idx,
                    "end_idx": len(all_examples),
                    "n_examples": len(examples),
                    "max_activation": examples[0].activation,
                    "mean_activation": (
                        stats["sum"] / stats["count"]
                        if stats["count"] > 0
                        else 0
                    ),
                    "total_activations": stats["count"],
                }
            )

        # Save consolidated examples
        logger.info(f"Saving {len(all_examples)} total examples...")
        examples_df = pl.DataFrame(all_examples)
        examples_df.write_parquet(
            self.output_dir / "all_examples.parquet",
            compression="zstd",
            compression_level=3,
        )

        # Save index
        logger.info(f"Saving index for {len(feature_index)} features...")
        index_df = pl.DataFrame(feature_index)
        index_df.write_parquet(
            self.output_dir / "feature_index.parquet", compression="zstd"
        )

        # Save metadata
        metadata = {
            "n_features": self.n_features,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "total_examples": len(all_examples),
            "features_with_examples": len(feature_index),
            "includes_negative_activations": self.include_negative,
        }

        with open(self.output_dir / "storage_metadata.json", "wb") as f:
            f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

        logger.info(f"✓ Storage created at {self.output_dir}")
        logger.info(f"  Total examples: {len(all_examples):,}")
        logger.info(f"  Features with examples: {len(feature_index):,}")
        logger.info(
            f"  Storage size: {(self.output_dir / 'all_examples.parquet').stat().st_size / (1024**2):.2f} MB"
        )


class OptimizedExampleReader:
    """Fast reader for the optimized storage format."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)

        # Load index
        self.index = pl.read_parquet(self.storage_dir / "feature_index.parquet")

        # Load all examples (lazy)
        self.examples = pl.scan_parquet(
            self.storage_dir / "all_examples.parquet"
        )

        # Create feature lookup
        self.feature_lookup = {
            row["feature_id"]: (row["start_idx"], row["end_idx"])
            for row in self.index.to_dicts()
        }

    def get_feature_examples(
        self, feature_id: int, limit: int = None
    ) -> pl.DataFrame:
        """Get examples for a feature."""

        if feature_id not in self.feature_lookup:
            return pl.DataFrame()

        start_idx, end_idx = self.feature_lookup[feature_id]

        # Use slice to get specific rows
        examples = self.examples.slice(start_idx, end_idx - start_idx)

        if limit:
            examples = examples.head(limit)

        # Collect and parse JSON
        df = examples.collect()

        # Parse ability tokens from JSON
        df = df.with_columns(
            pl.col("ability_tokens_json")
            .map_elements(
                lambda x: orjson.loads(x) if x else [],
                return_dtype=pl.List(pl.Int64),
            )
            .alias("ability_tokens")
        )

        return df

    def get_feature_stats(self, feature_id: int) -> dict:
        """Get statistics for a feature."""

        row = self.index.filter(pl.col("feature_id") == feature_id)

        if len(row) == 0:
            return None

        return row.to_dicts()[0]

    def get_top_features(self, n: int = 10) -> pl.DataFrame:
        """Get features with highest max activation."""

        return self.index.sort("max_activation", descending=True).head(n)


def main():
    """Build optimized storage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build optimized example storage for the dashboard."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/e/activations_ultra_efficient",
        help="Path to the activation data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/e/dashboard_examples_optimized",
        help="Path to output directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top examples to keep per feature",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Activation threshold",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Include negative activations (required for PageRank analysis)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    builder = OptimizedExampleStorage(
        data_dir=data_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        activation_threshold=args.threshold,
        include_negative=args.include_negative,
    )

    if args.include_negative:
        logger.info("Including negative activations (required for PageRank)")

    builder.process_all_batches()

    # Test reader
    print("\n" + "=" * 60)
    print("Testing reader...")

    reader = OptimizedExampleReader(output_dir)

    # Get top features
    top_features = reader.get_top_features(5)
    print(f"\nTop 5 features by max activation:")
    print(top_features.select(["feature_id", "max_activation", "n_examples"]))

    # Test getting examples
    if len(top_features) > 0:
        test_feature = top_features["feature_id"][0]
        examples = reader.get_feature_examples(test_feature, limit=3)
        print(f"\nTop 3 examples for feature {test_feature}:")
        print(examples.select(["activation", "weapon_id", "ability_tokens"]))


if __name__ == "__main__":
    main()
