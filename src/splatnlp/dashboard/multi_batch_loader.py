#!/usr/bin/env python
"""
Multi-batch loader for Ultra model activations.
Handles loading and combining multiple joblib files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class MultiBatchActivationLoader:
    """Load and manage activations from multiple batch files."""

    def __init__(self, batch_dir: Path, max_batches: Optional[int] = None):
        """
        Initialize the multi-batch loader.

        Args:
            batch_dir: Directory containing batch_*.joblib files
            max_batches: Maximum number of batches to load (None = all)
        """
        self.batch_dir = Path(batch_dir)
        self.max_batches = max_batches
        self.batches = []
        self.metadata = {}

        self._load_batches()

    def _load_batches(self):
        """Load all available batch files."""
        batch_files = sorted(self.batch_dir.glob("batch_*.joblib"))

        if self.max_batches:
            batch_files = batch_files[: self.max_batches]

        logger.info(f"Found {len(batch_files)} batch files")

        for batch_file in batch_files:
            logger.info(f"Loading {batch_file.name}...")
            try:
                batch_data = joblib.load(batch_file)
                self.batches.append(
                    {"file": batch_file.name, "data": batch_data}
                )

                # Extract metadata from first batch
                if not self.metadata and isinstance(batch_data, dict):
                    self.metadata = {
                        "sae_hidden_dim": batch_data.get("sae_hidden_dim"),
                        "embedding_dim": batch_data.get("embedding_dim"),
                        "num_features": batch_data.get("num_features"),
                    }
            except Exception as e:
                logger.error(f"Failed to load {batch_file}: {e}")

    def get_combined_activations(self) -> Dict[str, np.ndarray]:
        """
        Combine activations from all loaded batches.

        Returns:
            Dictionary with combined arrays for each activation type
        """
        if not self.batches:
            raise ValueError("No batches loaded")

        combined = {}

        # Get keys from first batch
        first_batch = self.batches[0]["data"]

        for key in first_batch.keys():
            if isinstance(first_batch[key], np.ndarray):
                # Combine arrays from all batches
                arrays = [batch["data"][key] for batch in self.batches]
                combined[key] = np.concatenate(arrays, axis=0)
                logger.info(f"Combined {key}: shape {combined[key].shape}")
            else:
                # Copy non-array data from first batch
                combined[key] = first_batch[key]

        return combined

    def get_batch(self, batch_idx: int) -> Dict[str, Any]:
        """Get a specific batch by index."""
        if batch_idx >= len(self.batches):
            raise IndexError(f"Batch {batch_idx} not found")
        return self.batches[batch_idx]["data"]

    def get_streaming_iterator(self, chunk_size: int = 1000):
        """
        Create an iterator that streams activations in chunks.

        Args:
            chunk_size: Number of examples per chunk

        Yields:
            Dictionary with chunk of activations
        """
        for batch in self.batches:
            batch_data = batch["data"]

            # Assume main data is in 'hidden_activations' key
            if "hidden_activations" in batch_data:
                activations = batch_data["hidden_activations"]
                n_examples = activations.shape[0]

                for i in range(0, n_examples, chunk_size):
                    end_idx = min(i + chunk_size, n_examples)
                    yield {
                        "hidden_activations": activations[i:end_idx],
                        "batch_file": batch["file"],
                        "chunk_start": i,
                        "chunk_end": end_idx,
                    }

    def save_combined(self, output_path: Path, compress: int = 3):
        """
        Save combined activations to a single file.

        Args:
            output_path: Path for output file
            compress: Compression level (0-9)
        """
        logger.info(f"Combining and saving to {output_path}...")
        combined = self.get_combined_activations()
        joblib.dump(combined, output_path, compress=compress)
        logger.info(f"Saved combined activations to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about loaded batches."""
        total_examples = 0
        total_size = 0

        for batch in self.batches:
            if "hidden_activations" in batch["data"]:
                total_examples += batch["data"]["hidden_activations"].shape[0]

            # Estimate memory size
            for value in batch["data"].values():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes

        return {
            "num_batches": len(self.batches),
            "total_examples": total_examples,
            "total_size_mb": total_size / (1024 * 1024),
            "metadata": self.metadata,
            "batch_files": [b["file"] for b in self.batches],
        }


def combine_batches_cli():
    """Command-line interface for combining batches."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine Ultra model activation batches"
    )
    parser.add_argument("batch_dir", help="Directory containing batch files")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--max-batches", "-m", type=int, help="Maximum batches to load"
    )
    parser.add_argument("--info", action="store_true", help="Show info only")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load batches
    loader = MultiBatchActivationLoader(
        Path(args.batch_dir), max_batches=args.max_batches
    )

    # Show summary
    summary = loader.get_summary()
    print("\n=== Batch Summary ===")
    print(f"Number of batches: {summary['num_batches']}")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")
    print(
        f"Files: {summary['batch_files'][:5]}..."
        if len(summary["batch_files"]) > 5
        else f"Files: {summary['batch_files']}"
    )

    if args.info:
        return

    # Save combined if output specified
    if args.output:
        loader.save_combined(Path(args.output))
        print(f"\n✓ Combined activations saved to {args.output}")


if __name__ == "__main__":
    combine_batches_cli()
