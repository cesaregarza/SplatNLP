#!/usr/bin/env python3
"""
Demo script showing how to efficiently analyze the converted activation data.
Uses Polars for fast queries and Zarr for streaming array access.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import zarr


class ActivationAnalyzer:
    """Efficient analyzer for the converted activation data."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"
        self.activations_dir = self.data_dir / "activations"
        self.embeddings_dir = self.data_dir / "embeddings"

        # Load conversion metadata
        with open(self.data_dir / "conversion_metadata.json") as f:
            self.meta = json.load(f)

        print(f"Loaded data with {self.meta['total_samples']:,} samples")

    def find_top_activated_features(self, n_features: int = 10) -> List[dict]:
        """Find the most frequently activated features across all data."""

        print(f"\nFinding top {n_features} most activated features...")

        # Track activation counts per feature
        feature_counts = np.zeros(24576)

        for batch_idx in range(self.meta["n_batches"]):
            # Load activations for this batch
            z = zarr.open(
                str(self.activations_dir / f"batch_{batch_idx:04d}.zarr"),
                mode="r",
            )

            # Process in chunks to manage memory
            for i in range(0, z.shape[0], 5000):
                chunk = z[i : min(i + 5000, z.shape[0])]
                # Count non-zero activations per feature
                feature_counts += (chunk > 0).sum(axis=0)

        # Get top features
        top_indices = np.argsort(feature_counts)[-n_features:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "feature_id": int(idx),
                    "activation_count": int(feature_counts[idx]),
                    "activation_rate": float(
                        feature_counts[idx] / self.meta["total_samples"]
                    ),
                }
            )

        return results

    def analyze_weapon_patterns(self, weapon_id: int, limit: int = 100) -> dict:
        """Analyze activation patterns for a specific weapon."""

        print(f"\nAnalyzing weapon_id={weapon_id}...")

        all_activations = []
        sample_count = 0

        for batch_idx in range(self.meta["n_batches"]):
            # Query metadata for this weapon
            df = (
                pl.scan_parquet(
                    self.metadata_dir / f"batch_{batch_idx:04d}.parquet"
                )
                .filter(pl.col("weapon_id_token") == weapon_id)
                .limit(limit - sample_count)
                .collect()
            )

            if len(df) == 0:
                continue

            # Get corresponding activations
            sample_ids = df["sample_id"].to_list()
            z = zarr.open(
                str(self.activations_dir / f"batch_{batch_idx:04d}.zarr"),
                mode="r",
            )
            activations = z[sample_ids]

            all_activations.append(activations)
            sample_count += len(df)

            if sample_count >= limit:
                break

        if not all_activations:
            return {"weapon_id": weapon_id, "samples_found": 0}

        # Combine all activations
        all_acts = np.vstack(all_activations)

        # Compute statistics
        mean_activation = all_acts.mean(axis=0)
        top_features = np.argsort(mean_activation)[-10:][::-1]

        return {
            "weapon_id": weapon_id,
            "samples_found": sample_count,
            "mean_sparsity": float((all_acts == 0).mean()),
            "top_features": [
                {
                    "feature_id": int(idx),
                    "mean_activation": float(mean_activation[idx]),
                }
                for idx in top_features
            ],
        }

    def find_similar_samples(
        self, batch_idx: int, sample_idx: int, n_similar: int = 5
    ) -> List[dict]:
        """Find samples with similar activation patterns."""

        print(
            f"\nFinding samples similar to batch_{batch_idx:04d}[{sample_idx}]..."
        )

        # Get reference activation
        z_ref = zarr.open(
            str(self.activations_dir / f"batch_{batch_idx:04d}.zarr"), mode="r"
        )
        ref_activation = z_ref[sample_idx]

        # Normalize for cosine similarity
        ref_norm = ref_activation / (np.linalg.norm(ref_activation) + 1e-8)

        similarities = []

        for b_idx in range(self.meta["n_batches"]):
            z = zarr.open(
                str(self.activations_dir / f"batch_{b_idx:04d}.zarr"), mode="r"
            )

            # Process in chunks
            for i in range(0, z.shape[0], 1000):
                chunk = z[i : min(i + 1000, z.shape[0])]

                # Compute cosine similarities
                norms = np.linalg.norm(chunk, axis=1) + 1e-8
                chunk_norm = chunk / norms[:, np.newaxis]
                sims = chunk_norm @ ref_norm

                # Track top similarities
                for j, sim in enumerate(sims):
                    if b_idx == batch_idx and i + j == sample_idx:
                        continue  # Skip self

                    similarities.append(
                        {
                            "batch_idx": b_idx,
                            "sample_idx": i + j,
                            "similarity": float(sim),
                        }
                    )

        # Sort and return top N
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Add metadata for top matches
        results = []
        for item in similarities[:n_similar]:
            # Load metadata for this sample
            df = pl.read_parquet(
                self.metadata_dir / f"batch_{item['batch_idx']:04d}.parquet"
            ).filter(pl.col("sample_id") == item["sample_idx"])

            if len(df) > 0:
                row = df.row(0, named=True)
                item["ability_tokens"] = row["ability_tokens"]
                item["weapon_id"] = row["weapon_id_token"]
                results.append(item)

        return results

    def compute_feature_sparsity_distribution(self) -> dict:
        """Compute the distribution of sparsity across features."""

        print("\nComputing feature sparsity distribution...")

        feature_activations = np.zeros(24576)

        for batch_idx in range(self.meta["n_batches"]):
            z = zarr.open(
                str(self.activations_dir / f"batch_{batch_idx:04d}.zarr"),
                mode="r",
            )

            # Count non-zero activations per feature
            for i in range(0, z.shape[0], 5000):
                chunk = z[i : min(i + 5000, z.shape[0])]
                feature_activations += (chunk > 0).sum(axis=0)

        # Compute sparsity (fraction of samples where feature is zero)
        sparsity = 1 - (feature_activations / self.meta["total_samples"])

        return {
            "mean_sparsity": float(sparsity.mean()),
            "std_sparsity": float(sparsity.std()),
            "min_sparsity": float(sparsity.min()),
            "max_sparsity": float(sparsity.max()),
            "ultra_sparse_features": int(
                (sparsity > 0.999).sum()
            ),  # >99.9% sparse
            "dead_features": int((sparsity == 1.0).sum()),  # Never activate
            "percentiles": {
                "25": float(np.percentile(sparsity, 25)),
                "50": float(np.percentile(sparsity, 50)),
                "75": float(np.percentile(sparsity, 75)),
                "95": float(np.percentile(sparsity, 95)),
                "99": float(np.percentile(sparsity, 99)),
            },
        }


def main():
    """Run demo analyses."""

    analyzer = ActivationAnalyzer(Path("/mnt/e/activations_ultra_efficient"))

    print("\n" + "=" * 80)
    print("DEMO ANALYSES")
    print("=" * 80)

    # 1. Find top activated features
    top_features = analyzer.find_top_activated_features(n_features=5)
    print("\nTop 5 Most Activated Features:")
    for feat in top_features:
        print(
            f"  Feature {feat['feature_id']}: "
            f"{feat['activation_count']:,} activations "
            f"({feat['activation_rate']:.2%} of samples)"
        )

    # 2. Analyze a specific weapon
    weapon_analysis = analyzer.analyze_weapon_patterns(weapon_id=5, limit=1000)
    print(f"\nWeapon ID 5 Analysis:")
    print(f"  Samples found: {weapon_analysis['samples_found']}")
    if weapon_analysis["samples_found"] > 0:
        print(f"  Mean sparsity: {weapon_analysis['mean_sparsity']:.4f}")
        print(f"  Top 3 features:")
        for feat in weapon_analysis["top_features"][:3]:
            print(
                f"    Feature {feat['feature_id']}: {feat['mean_activation']:.4f}"
            )

    # 3. Find similar samples
    similar = analyzer.find_similar_samples(
        batch_idx=0, sample_idx=0, n_similar=3
    )
    print(f"\nSamples similar to batch_0000[0]:")
    for item in similar:
        print(
            f"  Batch {item['batch_idx']}, Sample {item['sample_idx']}: "
            f"similarity={item['similarity']:.4f}, "
            f"weapon={item['weapon_id']}"
        )

    # 4. Compute sparsity distribution
    sparsity_dist = analyzer.compute_feature_sparsity_distribution()
    print(f"\nFeature Sparsity Distribution:")
    print(f"  Mean sparsity: {sparsity_dist['mean_sparsity']:.4f}")
    print(f"  Dead features: {sparsity_dist['dead_features']:,}")
    print(
        f"  Ultra-sparse (>99.9%): {sparsity_dist['ultra_sparse_features']:,}"
    )
    print(f"  Median sparsity: {sparsity_dist['percentiles']['50']:.4f}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
