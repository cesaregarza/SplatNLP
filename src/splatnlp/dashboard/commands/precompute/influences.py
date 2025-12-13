#!/usr/bin/env python3
"""
Precompute feature influences for all SAE features - Fixed version for Ultra model.

This script computes the influence of each SAE feature on output logits
and saves it in a format that can be quickly loaded by the dashboard.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Correct imports
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "ultra": {
        "embedding_dim": 32,
        "hidden_dim": 512,
        "num_layers": 3,
        "num_heads": 8,
        "num_inducing_points": 32,
        "dropout": 0.3,
        "sae_expansion_factor": 48.0,  # 512 * 48 = 24576
        "sae_features": 24576,
    },
    "full": {
        "embedding_dim": 512,
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "num_inducing_points": 32,
        "dropout": 0.1,
        "sae_expansion_factor": 4.0,  # 512 * 4 = 2048
        "sae_features": 2048,
    },
}


def detect_model_type(checkpoint_path: str) -> str:
    """Detect model type from checkpoint path."""
    if "super" in checkpoint_path or "ultra" in checkpoint_path:
        return "ultra"
    return "full"


def compute_influence_matrix(primary_model, sae_model):
    """
    Compute the influence matrix V × F where V is vocab size and F is number of features.

    This matrix shows how each SAE feature influences each output token.
    """
    # Get SAE decoder weights: [input_dim, hidden_dim]
    sae_decoder_weights = sae_model.decoder.weight.data.cpu()

    # Get output layer weights: [output_vocab_size, input_dim]
    output_weights = primary_model.output_layer.weight.data.cpu()

    # Compute influence: [vocab_size, hidden_dim]
    influence_matrix = torch.matmul(output_weights, sae_decoder_weights)

    logger.info(f"Influence matrix shape: {influence_matrix.shape}")

    return influence_matrix


def survey_feature_influences(
    influence_matrix, vocab, feature_labels=None, top_k=30, skip_special=True
):
    """
    Build a table that lists the top-k most positive and negative output-logit
    weights for every SAE feature.
    """
    VF = influence_matrix.cpu()
    V, F = VF.shape
    logger.info(f"Processing {F} features with {V} vocabulary tokens")

    inv_vocab = {v: k for k, v in vocab.items()}

    # Build mask to ignore specials
    if skip_special:
        specials = {"<PAD>", "<NULL>"}
        keep = torch.tensor(
            [inv_vocab.get(i, "") not in specials for i in range(V)]
        )
        VF = VF[keep]  # shape [V_keep, F]
        id_map = torch.arange(V)[keep]
    else:
        id_map = torch.arange(V)

    rows = []
    for fid in tqdm(range(F), desc="Processing features"):
        col = VF[:, fid]  # (V_keep,)

        # positive
        k_pos = min(top_k, col.numel())
        pos_vals, pos_idx = torch.topk(col, k_pos)

        # negative
        k_neg = min(top_k, col.numel())
        neg_vals, neg_idx = torch.topk(-col, k_neg)  # returns positive values

        label = ""
        if feature_labels:
            label = feature_labels.get(str(fid), {}).get("name", "")

        row = {"feature_id": fid, "feature_label": label}

        for r, (v, i) in enumerate(zip(pos_vals, pos_idx), 1):
            tok_id = int(id_map[i])
            row[f"+{r}_tok"] = inv_vocab.get(tok_id, f"Token_{tok_id}")
            row[f"+{r}_val"] = v.item()

        for r, (v, i) in enumerate(zip(neg_vals, neg_idx), 1):
            tok_id = int(id_map[i])
            row[f"-{r}_tok"] = inv_vocab.get(tok_id, f"Token_{tok_id}")
            row[f"-{r}_val"] = -v.item()  # restore sign

        rows.append(row)

    df = pd.DataFrame(rows).fillna("")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Precompute feature influences for SAE features"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=str,
        required=True,
        help="Path to SAE checkpoint",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to vocab JSON file",
    )
    parser.add_argument(
        "--weapon-vocab-path",
        type=str,
        required=True,
        help="Path to weapon vocab JSON file",
    )
    parser.add_argument(
        "--feature-labels",
        type=str,
        default=None,
        help="Path to feature labels JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for influence data",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top positive/negative influences to save",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "ultra", "full"],
        default="auto",
        help="Model type (auto-detect, ultra, or full)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Load vocabularies
    logger.info(
        f"Loading vocabs from {args.vocab_path} and {args.weapon_vocab_path}"
    )
    with open(args.vocab_path) as f:
        vocab = json.load(f)
    with open(args.weapon_vocab_path) as f:
        weapon_vocab = json.load(f)

    # Load feature labels if available
    feature_labels = None
    if args.feature_labels and Path(args.feature_labels).exists():
        logger.info(f"Loading feature labels from {args.feature_labels}")
        with open(args.feature_labels) as f:
            feature_labels = json.load(f)

    # Determine model type
    if args.model_type == "auto":
        model_type = detect_model_type(args.model_checkpoint)
        logger.info(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type

    config = MODEL_CONFIGS[model_type]

    # Initialize model
    logger.info(f"Loading {model_type} model from {args.model_checkpoint}")
    model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=len(vocab),
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_inducing_points=config["num_inducing_points"],
        use_layer_norm=True,
        dropout=config["dropout"],
        pad_token_id=vocab["<PAD>"],
    )

    # Load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()

    # Initialize SAE
    logger.info(f"Loading {model_type} SAE from {args.sae_checkpoint}")
    logger.info(
        f"SAE expansion factor: {config['sae_expansion_factor']} ({config['sae_features']} features)"
    )

    sae_model = SparseAutoencoder(
        input_dim=512,
        expansion_factor=config["sae_expansion_factor"],
        l1_coefficient=0.001,
        target_usage=0.05,
        usage_coeff=0.001,
    )

    # Load SAE checkpoint
    sae_checkpoint = torch.load(args.sae_checkpoint, map_location=args.device)
    if "model_state_dict" in sae_checkpoint:
        sae_model.load_state_dict(sae_checkpoint["model_state_dict"])
    else:
        sae_model.load_state_dict(sae_checkpoint)
    sae_model = sae_model.to(args.device)
    sae_model.eval()

    # Compute influence matrix
    logger.info("Computing influence matrix...")
    with torch.no_grad():
        influence_matrix = compute_influence_matrix(model, sae_model)

    # Verify dimensions
    expected_features = config["sae_features"]
    actual_features = influence_matrix.shape[1]
    if actual_features != expected_features:
        logger.warning(
            f"Feature count mismatch! Expected {expected_features}, got {actual_features}"
        )

    # Survey feature influences
    logger.info("Surveying feature influences...")
    influence_df = survey_feature_influences(
        influence_matrix=influence_matrix,
        vocab=vocab,
        feature_labels=feature_labels,
        top_k=args.top_k,
        skip_special=True,
    )

    # Save to file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        influence_df.to_parquet(output_path, index=False)
        logger.info(f"Saved influence data to {output_path} (Parquet format)")
    else:
        influence_df.to_csv(output_path, index=False)
        logger.info(f"Saved influence data to {output_path} (CSV format)")

    # Print summary statistics
    logger.info(f"\nSummary:")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Total features processed: {len(influence_df)}")
    logger.info(f"  Expected features: {expected_features}")
    labeled_features = influence_df[influence_df["feature_label"] != ""].shape[
        0
    ]
    logger.info(f"  Features with labels: {labeled_features}")

    # Show a few examples
    logger.info("\nExample feature influences:")
    sample_indices = [0, 100, 500, 1000, 5000, 10000, 20000]
    for idx in sample_indices:
        if idx < len(influence_df):
            row = influence_df.iloc[idx]
            label = row["feature_label"] or f"Feature {row['feature_id']}"
            top_pos = row["+1_tok"] if "+1_tok" in row else "N/A"
            top_neg = row["-1_tok"] if "-1_tok" in row else "N/A"
            logger.info(
                f"  Feature {idx:5d}: Top positive='{top_pos:20s}', Top negative='{top_neg:20s}'"
            )


if __name__ == "__main__":
    main()
