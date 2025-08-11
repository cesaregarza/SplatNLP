#!/usr/bin/env python3
"""
Unified precomputation script for SplatNLP dashboard.

Handles both Full and Ultra models, automatically detecting format and
computing appropriate data (influences, histograms, statistics).
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import zarr
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "ultra": {
        "model_path": "saved_models/dataset_v0_2_super/clean_slate.pth",
        "sae_path": "sae_runs/run_20250704_191557/sae_model_final.pth",
        "embedding_dim": 32,
        "hidden_dim": 512,
        "num_layers": 3,
        "num_heads": 8,
        "num_inducing_points": 32,
        "dropout": 0.3,
        "sae_expansion_factor": 48.0,
        "sae_features": 24576,
        "data_format": "efficient",  # Uses Zarr/Parquet
        "data_dir": "/mnt/e/activations_ultra_efficient",
    },
    "full": {
        "model_path": "saved_models/dataset_v0_2_full/model.pth",
        "sae_path": "saved_models/dataset_v0_2_full/sae_model.pth",
        "embedding_dim": 512,
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "num_inducing_points": 32,
        "dropout": 0.1,
        "sae_expansion_factor": 4.0,
        "sae_features": 2048,
        "data_format": "filesystem",  # Uses .npy files
        "neurons_root": "/mnt/e/activations2/outputs/neuron_acts",
        "meta_path": "/mnt/e/activations2/outputs/activations.metadata.joblib",
    },
}


class PrecomputeManager:
    """Manages all precomputation tasks."""

    def __init__(self, model_type: str, output_dir: Path, device: str = "cuda"):
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load vocabularies
        self.vocab = self._load_json(
            "saved_models/dataset_v0_2_full/vocab.json"
        )
        self.weapon_vocab = self._load_json(
            "saved_models/dataset_v0_2_full/weapon_vocab.json"
        )

        # Load feature labels if available
        labels_path = Path("src/splatnlp/dashboard/feature_labels.json")
        self.feature_labels = (
            self._load_json(labels_path) if labels_path.exists() else None
        )

        logger.info(f"Initialized {model_type} model precomputation")
        logger.info(f"Features: {self.config['sae_features']}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_json(self, path: str) -> dict:
        """Load JSON file."""
        with open(path) as f:
            return json.load(f)

    def compute_influences(self, top_k: int = 100) -> pd.DataFrame:
        """Compute feature influences on output tokens."""
        logger.info("=" * 60)
        logger.info("Computing Feature Influences")
        logger.info("=" * 60)

        # Initialize models
        model = self._init_primary_model()
        sae_model = self._init_sae_model()

        # Compute influence matrix
        logger.info("Computing influence matrix...")
        with torch.no_grad():
            sae_decoder = sae_model.decoder.weight.data.cpu()
            output_weights = model.output_layer.weight.data.cpu()
            influence_matrix = torch.matmul(output_weights, sae_decoder)

        logger.info(f"Influence matrix shape: {influence_matrix.shape}")

        # Survey influences
        logger.info(f"Processing {self.config['sae_features']} features...")
        influence_df = self._survey_influences(influence_matrix, top_k)

        # Save results
        output_path = self.output_dir / "influences.parquet"
        influence_df.to_parquet(output_path, index=False)
        logger.info(f"✓ Saved influence data to {output_path}")

        return influence_df

    def _init_primary_model(self) -> SetCompletionModel:
        """Initialize primary model."""
        logger.info(f"Loading model from {self.config['model_path']}")

        model = SetCompletionModel(
            vocab_size=len(self.vocab),
            weapon_vocab_size=len(self.weapon_vocab),
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            output_dim=len(self.vocab),
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            num_inducing_points=self.config["num_inducing_points"],
            use_layer_norm=True,
            dropout=self.config["dropout"],
            pad_token_id=self.vocab["<PAD>"],
        )

        checkpoint = torch.load(
            self.config["model_path"], map_location=self.device
        )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model.to(self.device).eval()

    def _init_sae_model(self) -> SparseAutoencoder:
        """Initialize SAE model."""
        logger.info(f"Loading SAE from {self.config['sae_path']}")

        sae_model = SparseAutoencoder(
            input_dim=512,
            expansion_factor=self.config["sae_expansion_factor"],
            l1_coefficient=0.001,
            target_usage=0.05,
            usage_coeff=0.001,
        )

        checkpoint = torch.load(
            self.config["sae_path"], map_location=self.device
        )
        if "model_state_dict" in checkpoint:
            sae_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            sae_model.load_state_dict(checkpoint)

        return sae_model.to(self.device).eval()

    def _survey_influences(
        self, influence_matrix: torch.Tensor, top_k: int
    ) -> pd.DataFrame:
        """Survey top positive and negative influences for each feature."""
        VF = influence_matrix.cpu()
        V, F = VF.shape

        inv_vocab = {v: k for k, v in self.vocab.items()}

        # Filter special tokens
        specials = {"<PAD>", "<NULL>"}
        keep = torch.tensor(
            [inv_vocab.get(i, "") not in specials for i in range(V)]
        )
        VF = VF[keep]
        id_map = torch.arange(V)[keep]

        rows = []
        for fid in tqdm(range(F), desc="Processing features"):
            col = VF[:, fid]

            # Get top positive and negative
            pos_vals, pos_idx = torch.topk(col, min(top_k, col.numel()))
            neg_vals, neg_idx = torch.topk(-col, min(top_k, col.numel()))

            row = {"feature_id": fid, "feature_label": ""}
            if self.feature_labels:
                row["feature_label"] = self.feature_labels.get(
                    str(fid), {}
                ).get("name", "")

            for r, (v, i) in enumerate(zip(pos_vals, pos_idx), 1):
                tok_id = int(id_map[i])
                row[f"+{r}_tok"] = inv_vocab.get(tok_id, f"Token_{tok_id}")
                row[f"+{r}_val"] = v.item()

            for r, (v, i) in enumerate(zip(neg_vals, neg_idx), 1):
                tok_id = int(id_map[i])
                row[f"-{r}_tok"] = inv_vocab.get(tok_id, f"Token_{tok_id}")
                row[f"-{r}_val"] = -v.item()

            rows.append(row)

        return pd.DataFrame(rows).fillna("")

    def compute_histograms(self, nb_bins: int = 100) -> Dict:
        """Compute histograms based on data format."""
        logger.info("=" * 60)
        logger.info("Computing Histograms")
        logger.info("=" * 60)

        if self.config["data_format"] == "efficient":
            return self._compute_histograms_zarr(nb_bins)
        else:
            return self._compute_histograms_filesystem(nb_bins)

    def _compute_histograms_zarr(self, nb_bins: int) -> Dict:
        """Compute histograms from Zarr format (Ultra model) using parallel processing."""
        data_dir = Path(self.config["data_dir"])

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return {}

        # Import parallel histogram computation
        from splatnlp.dashboard.commands.precompute.histograms_parallel import (
            compute_histograms_parallel,
        )

        # Load metadata
        meta_path = data_dir / "conversion_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        n_batches = metadata.get("n_batches", 0)
        n_features = self.config["sae_features"]

        logger.info(
            f"Processing {n_features} features from {n_batches} batches (parallel)"
        )

        # Use parallel computation
        import multiprocessing

        n_workers = min(multiprocessing.cpu_count(), 16)

        histograms, stats = compute_histograms_parallel(
            data_dir=data_dir,
            n_features=n_features,
            n_batches=n_batches,
            nb_bins=nb_bins,
            n_workers=n_workers,
            chunk_size=100,  # Process 100 features per chunk
        )

        # Save results
        self._save_histograms(histograms, stats)
        return histograms

    def _compute_histograms_filesystem(self, nb_bins: int) -> Dict:
        """Compute histograms from filesystem format (Full model)."""
        neurons_root = Path(self.config["neurons_root"])

        if not neurons_root.exists():
            logger.warning(f"Neurons directory not found: {neurons_root}")
            return {}

        neuron_dirs = sorted(
            [
                d
                for d in neurons_root.iterdir()
                if d.is_dir() and d.name.startswith("neuron_")
            ]
        )

        logger.info(f"Processing {len(neuron_dirs)} neuron directories")

        histograms = {}
        stats = {}

        for neuron_dir in tqdm(neuron_dirs, desc="Computing histograms"):
            feature_id = int(neuron_dir.name.split("_")[1])

            acts_path = neuron_dir / "acts.npy"
            if acts_path.exists():
                acts = np.load(acts_path).astype(np.float32)
                counts, bins = np.histogram(acts, bins=nb_bins)

                histograms[feature_id] = {
                    "counts": counts.tolist(),
                    "lower_bounds": bins[:-1].tolist(),
                    "upper_bounds": bins[1:].tolist(),
                }

                stats[feature_id] = {
                    "min": float(acts.min()),
                    "max": float(acts.max()),
                    "mean": float(acts.mean()),
                    "std": float(acts.std()),
                    "n_non_zero": len(acts),
                }

        # Save results
        self._save_histograms(histograms, stats)
        return histograms

    def _save_histograms(self, histograms: Dict, stats: Dict):
        """Save histogram and statistics data."""
        # Save histograms
        hist_path = self.output_dir / "histograms.pkl"
        with open(hist_path, "wb") as f:
            pickle.dump(histograms, f)
        logger.info(f"✓ Saved histograms to {hist_path}")

        # Save statistics
        if stats:
            stats_path = self.output_dir / "feature_stats.pkl"
            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)
            logger.info(f"✓ Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified precomputation for SplatNLP dashboard"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ultra", "full"],
        default="ultra",
        help="Model type to precompute for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/precomputed_{model_type})",
    )
    parser.add_argument(
        "--skip-influences",
        action="store_true",
        help="Skip influence computation",
    )
    parser.add_argument(
        "--skip-histograms",
        action="store_true",
        help="Skip histogram computation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top positive/negative influences to compute (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/precomputed_{args.model_type}")

    # Print configuration
    print("\n" + "=" * 60)
    print(f"SplatNLP Precomputation - {args.model_type.upper()} Model")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Top-k influences: {args.top_k}")
    print(f"Skip influences: {args.skip_influences}")
    print(f"Skip histograms: {args.skip_histograms}")
    print()

    # Initialize manager
    manager = PrecomputeManager(args.model_type, output_dir, args.device)

    # Run precomputation tasks
    if not args.skip_influences:
        manager.compute_influences(top_k=args.top_k)

    if not args.skip_histograms:
        manager.compute_histograms()

    # Summary
    print("\n" + "=" * 60)
    print("Precomputation Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("\nTo run the dashboard:")
    print(f"  ./run_dashboard.sh --model-type {args.model_type}")


if __name__ == "__main__":
    main()
