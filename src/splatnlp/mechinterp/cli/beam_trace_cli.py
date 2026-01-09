"""CLI for generating beam search traces with SAE feature activations.

This script runs beam search from <NULL> to completion, capturing SAE feature
activations at each step. Supports both Full (2K features) and Ultra (24K features)
models to enable feature splitting comparison.

Usage:
    poetry run python -m splatnlp.mechinterp.cli.beam_trace_cli \
        --weapon-id weapon_id_1000 \
        --model ultra \
        --output tmp_results/stamper_trace.json \
        --attach-labels
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.hooks import register_hooks
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.utils.infer import build_predict_abilities
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import reconstruct_build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Model and SAE paths for each model type
MODEL_PATHS = {
    "ultra": {
        "model": Path(
            "/mnt/e/dev_spillover/SplatNLP/saved_models/dataset_v0_2_super/clean_slate.pth"
        ),
        "sae": Path(
            "/mnt/e/dev_spillover/SplatNLP/sae_runs/run_20250704_191557/sae_model_final.pth"
        ),
        "labels": Path("/mnt/e/mechinterp_runs/labels/consolidated_ultra.json"),
        "n_features": 24576,
        "expansion_factor": 48.0,
    },
    "full": {
        "model": Path(
            "/root/dev/SplatNLP/saved_models/dataset_v0_2_full/model.pth"
        ),
        "sae": Path(
            "/root/dev/SplatNLP/saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_model_final.pth"
        ),
        "labels": Path("/mnt/e/mechinterp_runs/labels/consolidated_full.json"),
        "n_features": 2048,
        "expansion_factor": 4.0,
    },
}

# Shared vocab paths
VOCAB_PATH = Path(
    "/root/dev/SplatNLP/saved_models/dataset_v0_2_full/vocab.json"
)
WEAPON_VOCAB_PATH = Path(
    "/root/dev/SplatNLP/saved_models/dataset_v0_2_full/weapon_vocab.json"
)

# Model architecture (same for both Full and Ultra)
MODEL_CONFIG = {
    "embedding_dim": 32,
    "hidden_dim": 512,
    "num_layers": 3,
    "num_heads": 8,
    "use_layer_norm": True,
}


def load_vocabs() -> tuple[dict[str, int], dict[str, int]]:
    """Load vocabulary files."""
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    with open(WEAPON_VOCAB_PATH) as f:
        weapon_vocab = json.load(f)
    return vocab, weapon_vocab


def load_model(
    model_type: Literal["full", "ultra"],
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    device: torch.device,
) -> SetCompletionModel:
    """Load the SetCompletionModel for the specified type."""
    paths = MODEL_PATHS[model_type]

    model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=MODEL_CONFIG["embedding_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        output_dim=len(vocab),
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        use_layer_norm=MODEL_CONFIG["use_layer_norm"],
        pad_token_id=vocab["<PAD>"],
    )

    state_dict = torch.load(
        paths["model"], map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded {model_type} model from {paths['model']}")
    return model


def load_sae(
    model_type: Literal["full", "ultra"],
    device: torch.device,
) -> SparseAutoencoder:
    """Load the SAE for the specified model type."""
    paths = MODEL_PATHS[model_type]

    sae = SparseAutoencoder(
        input_dim=MODEL_CONFIG["hidden_dim"],
        expansion_factor=paths["expansion_factor"],
    )

    checkpoint = torch.load(
        paths["sae"], map_location=device, weights_only=True
    )

    # Load weights directly to handle extra keys in checkpoint
    sae.encoder.weight.data = checkpoint["encoder.weight"]
    sae.encoder.bias.data = checkpoint["encoder.bias"]
    sae.decoder.weight.data = checkpoint["decoder.weight"]
    if "decoder.bias" in checkpoint:
        sae.decoder.bias.data = checkpoint["decoder.bias"]

    sae.to(device)
    sae.eval()

    logger.info(f"Loaded {model_type} SAE from {paths['sae']}")
    return sae


def load_labels(model_type: Literal["full", "ultra"]) -> dict[int, str]:
    """Load feature labels from consolidated JSON."""
    paths = MODEL_PATHS[model_type]
    labels_path = paths["labels"]

    if not labels_path.exists():
        logger.warning(f"Labels file not found: {labels_path}")
        return {}

    with open(labels_path) as f:
        data = json.load(f)

    # Extract labels - handle both formats
    labels = {}
    for key, value in data.items():
        try:
            fid = int(key)
            if isinstance(value, dict):
                # New format with research_label or dashboard_name
                label = value.get("research_label") or value.get(
                    "dashboard_name", f"Feature {fid}"
                )
            else:
                label = str(value)
            labels[fid] = label
        except (ValueError, TypeError):
            continue

    logger.info(f"Loaded {len(labels)} labels for {model_type}")
    return labels


def create_predict_fn_with_sae(
    model: SetCompletionModel,
    sae: SparseAutoencoder,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    device: torch.device,
):
    """Create predict functions that return probabilities and SAE activations.

    Returns a single-example ``predict_fn`` and a batched ``predict_batch_fn``.
    Both are compatible with ``beam_search``'s ``record_traces`` feature.
    """
    # Register hook to capture SAE activations
    hook, handle = register_hooks(model, sae, bypass=False, no_change=True)

    inv_vocab = {v: k for k, v in vocab.items()}
    pad_id = vocab["<PAD>"]

    def predict_fn(tokens: list[str], weapon_id: str):
        """Predict probabilities and return SAE activations."""
        input_tokens = [vocab[t] for t in tokens]
        input_tensor = torch.tensor(input_tokens, device=device).unsqueeze(0)
        weapon_tensor = torch.tensor(
            [weapon_vocab[weapon_id]], device=device
        ).unsqueeze(0)
        key_padding_mask = (input_tensor == vocab["<PAD>"]).to(device)

        with torch.no_grad():
            outputs = model(
                input_tensor, weapon_tensor, key_padding_mask=key_padding_mask
            )
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Get activations from hook
        activations = None
        if hook.last_h_post is not None:
            activations = hook.last_h_post.squeeze().cpu().numpy()

        # Convert probs to dict
        probs_dict = {inv_vocab[i]: float(probs[i]) for i in range(len(probs))}

        return probs_dict, activations

    def predict_batch_fn(token_batches: list[list[str]], weapon_id: str):
        """Predict probabilities and activations for a batch of contexts."""
        if not token_batches:
            return [], []

        batch_size = len(token_batches)
        max_len = max(len(toks) for toks in token_batches)
        input_tokens = torch.full(
            (batch_size, max_len),
            pad_id,
            device=device,
            dtype=torch.long,
        )
        for row_idx, toks in enumerate(token_batches):
            if not toks:
                continue
            token_ids = [vocab[t] for t in toks]
            input_tokens[row_idx, : len(token_ids)] = torch.tensor(
                token_ids,
                device=device,
                dtype=torch.long,
            )

        weapon_tensor = torch.full(
            (batch_size, 1),
            weapon_vocab[weapon_id],
            device=device,
            dtype=torch.long,
        )
        key_padding_mask = input_tokens == pad_id

        with torch.no_grad():
            outputs = model(
                input_tokens,
                weapon_tensor,
                key_padding_mask=key_padding_mask,
            )
            probs = torch.sigmoid(outputs).detach().cpu().numpy()

        probs_batch = [
            {inv_vocab[i]: float(row[i]) for i in range(row.shape[0])}
            for row in probs
        ]

        activations_batch = [None] * batch_size
        if hook.last_h_post is not None:
            acts = hook.last_h_post.detach().cpu().numpy()
            activations_batch = [acts[i] for i in range(batch_size)]

        return probs_batch, activations_batch

    return predict_fn, predict_batch_fn, handle


def format_trace_summary(
    traces: list,
    top_k_features: int,
    labels: dict[int, str],
    top_k_preds: int = 8,
) -> list[dict]:
    """Format trace frames into summary format matching existing traces."""
    summary = []

    for frame in traces:
        # Get top predictions
        sorted_preds = sorted(frame.logits.items(), key=lambda x: -x[1])[
            :top_k_preds
        ]
        top_preds = [[tok, prob] for tok, prob in sorted_preds]

        # Get top features
        top_features = []
        if frame.activations is not None:
            acts = np.array(frame.activations)
            top_indices = np.argsort(acts)[-top_k_features:][::-1]
            for idx in top_indices:
                fid = int(idx)
                activation = float(acts[idx])
                label = labels.get(fid, f"Feature {fid}")
                top_features.append(
                    {
                        "feature_id": fid,
                        "activation": activation,
                        "label": label,
                    }
                )

        # Get capstones (sorted for consistency)
        capstones = sorted(frame.partial_caps.keys())

        # Determine what was added this step
        if summary:
            prev_capstones = set(summary[-1]["capstones"])
            added = [c for c in capstones if c not in prev_capstones]
        else:
            added = capstones

        summary.append(
            {
                "step": frame.step,
                "beam_rank": frame.beam_rank,
                "capstones": capstones,
                "added_this_step": added,
                "top_preds": top_preds,
                "top_features": top_features,
            }
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate beam search traces with SAE feature activations"
    )
    parser.add_argument(
        "--weapon-id",
        type=str,
        required=True,
        help="Weapon token (e.g., weapon_id_1000 for Splatana Stamper)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["full", "ultra"],
        default="ultra",
        help="Model type to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--initial-context",
        type=str,
        default="",
        help="Initial tokens (comma-separated)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam width",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum beam search steps",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=10,
        help="Number of top features to include per step",
    )
    parser.add_argument(
        "--attach-labels",
        action="store_true",
        help="Attach labels from consolidated JSON",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load vocabs
    logger.info("Loading vocabularies...")
    vocab, weapon_vocab = load_vocabs()

    # Validate weapon ID
    if args.weapon_id not in weapon_vocab:
        logger.error(f"Unknown weapon ID: {args.weapon_id}")
        logger.info(f"Available weapons: {list(weapon_vocab.keys())[:10]}...")
        return

    # Load model and SAE
    logger.info(f"Loading {args.model} model...")
    model = load_model(args.model, vocab, weapon_vocab, device)

    logger.info(f"Loading {args.model} SAE...")
    sae = load_sae(args.model, device)

    # Load labels if requested
    labels = {}
    if args.attach_labels:
        labels = load_labels(args.model)

    # Create predict function
    logger.info("Creating prediction function with SAE hooks...")
    predict_fn, predict_batch_fn, handle = create_predict_fn_with_sae(
        model, sae, vocab, weapon_vocab, device
    )

    # Parse initial context
    initial_context = []
    if args.initial_context:
        initial_context = [t.strip() for t in args.initial_context.split(",")]
        logger.info(f"Starting with initial context: {initial_context}")

    # Run beam search with traces
    logger.info(
        f"Running beam search (beam_size={args.beam_size}, "
        f"max_steps={args.max_steps})..."
    )
    allocator = Allocator()

    result = reconstruct_build(
        predict_fn=predict_fn,
        predict_batch_fn=predict_batch_fn,
        weapon_id=args.weapon_id,
        initial_context=initial_context,
        allocator=allocator,
        beam_size=args.beam_size,
        max_steps=args.max_steps,
        top_k=1,
        record_traces=True,
    )

    # Clean up hook
    handle.remove()

    if result is None or result[0] is None:
        logger.error("No valid build could be constructed")
        return

    builds, traces = result
    build = builds[0]
    trace = traces[0]

    logger.info(f"Build completed with {build.total_ap} AP")

    # Format output
    output = {
        "model_type": args.model,
        "weapon_id": args.weapon_id,
        "build": {
            "mains": build.mains,
            "subs": dict(build.subs),
            "total_ap": build.total_ap,
            "achieved_ap": dict(build.achieved_ap),
        },
        "trace_summary": format_trace_summary(
            trace, args.top_k_features, labels
        ),
    }

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Trace saved to {output_path}")
    logger.info(f"Trace has {len(output['trace_summary'])} steps")


if __name__ == "__main__":
    main()
