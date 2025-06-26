"""
Train a Sparse Auto-Encoder (SAE) on SetCompletionModel activations.

Key improvements vs. the old CLI
--------------------------------
* Cleaner argument list with sensible defaults.
* No runtime path hacking - relies on the installed `splatnlp` package.
* Consistent logging (to console **and** file).
* Robust estimation of `T_max` for the cosine LR scheduler.
* Passes `val_loader` into `train_sae_model`, so validation happens during
  training.
* Proper JSON/ORJSON serialisation of metrics and run-config.
* Optional gradient clipping, KL-warm-up schedule, dead-neuron resampling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Package imports
# (all live inside the installed `splatnlp` package – no sys.path hacking)
# ──────────────────────────────────────────────────────────────────────────────
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.data_objects import SAEConfig
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.monosemantic_sae.sae_training import (
    evaluate_sae_model,
    train_sae_model,
)
from splatnlp.monosemantic_sae.utils import (
    load_json_from_path,
    load_tokenized_data,
    setup_hook,
)
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)
from splatnlp.utils.convert_ddp_state import convert_ddp_state

# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_ROOT_LOGGER = logging.getLogger()
_LOGGER = logging.getLogger(__name__)


def _setup_logging(save_dir: Path, verbose: bool) -> None:
    """Configure root logger once per run."""
    log_level = logging.INFO if verbose else logging.WARNING
    log_file = save_dir / "sae_training.log"

    # Re-initialise handlers - prevents duplicated log lines in Jupyter
    for h in _ROOT_LOGGER.handlers[:]:
        _ROOT_LOGGER.removeHandler(h)

    logging.basicConfig(
        level=log_level,
        format=_LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        force=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a Sparse Auto-Encoder on SetCompletionModel activations.",
    )

    # ── Required paths ───────────────────────────────────────────────────────
    ap.add_argument(
        "--model-checkpoint",
        required=True,
        type=str,
        help="Path/URL to the primary SetCompletionModel .pth checkpoint.",
    )
    ap.add_argument(
        "--data-csv",
        required=True,
        type=str,
        help="Path to the tokenised dataset TSV (abilities, weapons, ...).",
    )
    ap.add_argument(
        "--vocab-path",
        required=True,
        type=str,
        help="Path/URL to the ability vocabulary JSON.",
    )
    ap.add_argument(
        "--weapon-vocab-path",
        required=True,
        type=str,
        help="Path/URL to the weapon vocabulary JSON.",
    )
    ap.add_argument(
        "--save-dir",
        required=True,
        type=str,
        help="Directory where models / metrics will be written.",
    )

    # ── Hook options ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--hook-target",
        choices=["masked_mean", "token_ff"],
        default="masked_mean",
        help="Activation point to hijack for the SAE.",
    )
    ap.add_argument(
        "--hook-layer-index",
        type=int,
        help="(token_ff) Transformer layer index.",
    )
    ap.add_argument(
        "--hook-ff-module-index",
        type=int,
        help="(token_ff) Feed-forward sub-module index in that layer.",
    )

    # ── Primary model hyper-params (only shape-relevant) ─────────────────────
    ap.add_argument("--primary-embedding-dim", type=int, default=32)
    ap.add_argument(
        "--primary-hidden-dim",
        type=int,
        default=512,
        help="Dimension of the masked-mean vector - becomes SAE input.",
    )
    ap.add_argument("--primary-num-layers", type=int, default=3)
    ap.add_argument("--primary-num-heads", type=int, default=8)
    ap.add_argument("--primary-num-inducing", type=int, default=32)
    ap.add_argument(
        "--primary-use-layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--primary-dropout", type=float, default=0.0)

    # ── SAE hyper‑params ─────────────────────────────────────────────────────
    ap.add_argument(
        "--expansion-factor",
        type=float,
        default=4.0,
        help="Hidden / input dimension ratio.",
    )
    ap.add_argument("--l1-coeff", type=float, default=5e-6)
    ap.add_argument("--target-usage", type=float, default=0.05)
    ap.add_argument(
        "--usage-coeff",
        type=float,
        default=0.0,
        help="Base coefficient for the KL usage term (0 ⇒ disabled).",
    )
    ap.add_argument("--dead-neuron-threshold", type=float, default=1e-8)

    # KL schedule
    ap.add_argument("--kl-warmup-steps", type=int, default=6_000)
    ap.add_argument("--kl-period-steps", type=int, default=60_000)
    ap.add_argument("--kl-floor", type=float, default=0.05)

    # ── Training loop ────────────────────────────────────────────────────────
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--primary-batch-size",
        type=int,
        default=32,
        help="Batch size for the *primary* model forward pass.",
    )
    ap.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="Circular buffer size for captured activations.",
    )
    ap.add_argument(
        "--sae-batch-size",
        type=int,
        default=1024,
        help="Batch size for SAE optimisation steps.",
    )
    ap.add_argument(
        "--steps-before-train",
        type=int,
        default=50_000,
        help="#activations in buffer before SAE updates start.",
    )
    ap.add_argument(
        "--sae-train-steps",
        type=int,
        default=4,
        help="SAE steps per primary forward (after warm-up).",
    )
    ap.add_argument("--gradient-clip-val", type=float, default=1.0)
    ap.add_argument("--primary-data-fraction", type=float, default=0.005)
    ap.add_argument(
        "--num-workers", type=int, default=min(8, (os.cpu_count() or 8) // 2)
    )
    # Dead‑neuron resampling
    ap.add_argument(
        "--resample-steps",
        type=int,
        nargs="+",
        default=[7_000, 14_000, 21_000, 28_000],
    )
    ap.add_argument("--resample-weight", type=float, default=0.01)
    ap.add_argument("--resample-bias", type=float, default=-1.0)

    # ── Misc ────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="More logging + tqdm bars.",
    )

    return ap


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _json_write(path: Path, obj: Any) -> None:
    try:
        with open(path, "wb") as f:
            f.write(
                orjson.dumps(
                    obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
                )
            )
    except Exception:
        # Fall back to stdlib json for weird types
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _build_arg_parser().parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(save_dir, args.verbose)
    _LOGGER.info("Arguments:\n%s", json.dumps(vars(args), indent=2))

    device = torch.device(args.device)

    # -------------------------------------------------------------------- #
    # 1. Load vocabularies
    # -------------------------------------------------------------------- #
    vocab = load_json_from_path(args.vocab_path)
    weapon_vocab = load_json_from_path(args.weapon_vocab_path)
    pad_id = vocab.get("<PAD>")
    if pad_id is None:
        raise ValueError("'<PAD>' token missing from vocabulary.")
    _LOGGER.info(
        "Loaded vocab → %d tokens, weapon vocab → %d tokens.",
        len(vocab),
        len(weapon_vocab),
    )

    # -------------------------------------------------------------------- #
    # 2. Instantiate + load primary model
    # -------------------------------------------------------------------- #

    primary_model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=args.primary_embedding_dim,
        hidden_dim=args.primary_hidden_dim,
        output_dim=len(vocab),
        num_layers=args.primary_num_layers,
        num_heads=args.primary_num_heads,
        num_inducing_points=args.primary_num_inducing,
        use_layer_norm=args.primary_use_layernorm,
        dropout=args.primary_dropout,
        pad_token_id=pad_id,
    ).to(device)

    _LOGGER.info("Loading primary checkpoint from: %s", args.model_checkpoint)
    if args.model_checkpoint.startswith(("http://", "https://")):
        state_dict = torch.hub.load_state_dict_from_url(
            args.model_checkpoint, map_location=device
        )
    else:
        state_dict = torch.load(args.model_checkpoint, map_location=device)
    try:
        primary_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        _LOGGER.warning("Converting DDP state dict to non-DDP state dict")
        state_dict = convert_ddp_state(state_dict)
        primary_model.load_state_dict(state_dict, strict=True)
    primary_model.eval()

    # -------------------------------------------------------------------- #
    # 3. Register activation hook
    # -------------------------------------------------------------------- #
    hook, handle = setup_hook(
        primary_model,
        target=args.hook_target,
        layer_index=args.hook_layer_index,
        feedforward_module_index=args.hook_ff_module_index,
    )

    # -------------------------------------------------------------------- #
    # 4. Dataset → train/val/test DataLoaders
    # -------------------------------------------------------------------- #
    df = load_tokenized_data(args.data_csv)
    train_df, val_df, test_df = generate_tokenized_datasets(
        df, frac=args.primary_data_fraction
    )
    _LOGGER.info(
        "Dataset sizes → train %d | val %d | test %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_dl, val_dl, test_dl = generate_dataloaders(
        train_df,
        val_df,
        test_df,
        vocab_size=len(vocab),
        pad_token_id=pad_id,
        batch_size=args.primary_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # -------------------------------------------------------------------- #
    # 5. Build SAE & optimiser
    # -------------------------------------------------------------------- #
    sae = SparseAutoencoder(
        input_dim=args.primary_hidden_dim,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coeff,
        target_usage=args.target_usage,
        usage_coeff=args.usage_coeff,
        dead_neuron_threshold=args.dead_neuron_threshold,
        dead_neuron_steps=1,  # not used - resampling handled outside
    ).to(device)

    sae_config = SAEConfig(
        input_dim=args.primary_hidden_dim,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coeff,
        learning_rate=args.lr,
        resample_interval=999_999_999,  # turns off internal resample
        dead_neuron_threshold=args.dead_neuron_threshold,
        target_usage=args.target_usage,
        usage_coeff=args.usage_coeff,
    )

    optimiser = AdamW(sae.parameters(), lr=args.lr)

    # Cosine LR schedule - T_max = estimated total SAE updates
    batches_per_epoch = len(train_dl)
    prim_steps_total = args.epochs * batches_per_epoch
    prim_steps_warmup = args.steps_before_train // max(
        args.primary_batch_size, 1
    )
    prim_steps_with_sae = max(0, prim_steps_total - prim_steps_warmup)
    sae_updates_est = prim_steps_with_sae * args.sae_train_steps
    if sae_updates_est <= 0:
        sae_updates_est = prim_steps_total  # reasonable fallback
    _LOGGER.info(
        "CosineAnnealingLR: T_max = %d SAE updates (est.)", sae_updates_est
    )

    scheduler = CosineAnnealingLR(
        optimiser, T_max=sae_updates_est, eta_min=args.lr * 0.1
    )

    # -------------------------------------------------------------------- #
    # 6. Persist run configuration
    # -------------------------------------------------------------------- #
    run_cfg_path = save_dir / "sae_run_config.json"
    _json_write(
        run_cfg_path,
        {
            **vars(args),
            "sae_input_dim": sae.input_dim,
            "sae_hidden_dim": sae.hidden_dim,
        },
    )
    _LOGGER.info("Saved run-config → %s", run_cfg_path)

    # -------------------------------------------------------------------- #
    # 7. Train
    # -------------------------------------------------------------------- #
    try:
        metrics_history = train_sae_model(
            primary_model,
            sae,
            optimiser,
            scheduler,
            hook,
            sae_config,
            data_loader=train_dl,
            val_loader=val_dl,
            vocab=vocab,
            device=device,
            num_epochs=args.epochs,
            activation_buffer_size=args.buffer_size,
            sae_batch_size=args.sae_batch_size,
            steps_before_sae_train=args.steps_before_train,
            sae_train_steps_per_primary_step=args.sae_train_steps,
            resample_steps=set(args.resample_steps),
            resample_weight=args.resample_weight,
            resample_bias=args.resample_bias,
            kl_warmup_steps=args.kl_warmup_steps,
            kl_period_steps=args.kl_period_steps,
            kl_floor=args.kl_floor,
            log_interval=500,
            gradient_clip_val=args.gradient_clip_val,
            verbose=args.verbose,
        )
    finally:
        handle.remove()  # ensure we always detach the hook

    # -------------------------------------------------------------------- #
    # 8. Save SAE + metrics
    # -------------------------------------------------------------------- #
    sae_path = save_dir / "sae_model_final.pth"
    torch.save(sae.state_dict(), sae_path)
    _LOGGER.info("Saved SAE → %s", sae_path)

    if metrics_history:
        _json_write(save_dir / "sae_metrics_history.json", metrics_history)

    # -------------------------------------------------------------------- #
    # 9. Final evaluation on the held-out test set
    # -------------------------------------------------------------------- #
    _LOGGER.info("Running final SAE evaluation on test split …")
    test_hook, test_handle = setup_hook(
        primary_model,
        target=args.hook_target,
        layer_index=args.hook_layer_index,
        feedforward_module_index=args.hook_ff_module_index,
    )
    try:
        test_metrics = evaluate_sae_model(
            primary_model,
            sae,
            test_hook,
            test_dl,
            device,
            sae_config,
            vocab,
            description="Test",
        )
    finally:
        test_handle.remove()

    _json_write(save_dir / "sae_test_metrics.json", test_metrics)
    _LOGGER.info("Done - all artifacts saved to %s", save_dir)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
