import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import orjson
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.data_objects import SAEConfig
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.monosemantic_sae.sae_training.evaluate import evaluate_sae_model
from splatnlp.monosemantic_sae.sae_training.train import train_sae_model
from splatnlp.monosemantic_sae.utils import (
    load_json_from_path,
    load_tokenized_data,
    setup_hook,
)
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)


def setup_logging(log_file="sae_training.log", level=logging.INFO):
    """Sets up logging to stdout and a file."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        force=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train a Sparse Autoencoder on SetCompletionModel activations."
        )
    )

    # --- Arguments ---
    # Required Paths
    ap.add_argument(
        "--model-checkpoint",
        required=True,
        type=str,
        help="Path/URL to the primary SetCompletionModel checkpoint (.pth).",
    )
    ap.add_argument(
        "--data-csv",
        required=True,
        type=str,
        help="Path to the tokenized data CSV file.",
    )
    ap.add_argument(
        "--save-dir",
        required=True,
        type=str,
        help="Directory to save the trained SAE model and config.",
    )
    ap.add_argument(
        "--vocab-path",
        required=True,
        type=str,
        help="Path/URL to the vocabulary JSON file.",
    )
    ap.add_argument(
        "--weapon-vocab-path",
        required=True,
        type=str,
        help="Path/URL to the weapon vocabulary JSON file.",
    )

    # Model/Hook parameters
    ap.add_argument(
        "--hook-target",
        type=str,
        default="masked_mean",
        choices=["masked_mean", "token_ff"],
        help=(
            "Target for activation hook ('masked_mean' for input to "
            "output_layer, 'token_ff' for feed-forward activations)."
        ),
    )
    ap.add_argument(
        "--hook-layer-index",
        type=int,
        default=None,
        help="Layer index for 'token_ff' hook target.",
    )
    ap.add_argument(
        "--hook-ff-module-index",
        type=int,
        default=None,
        help=(
            "Feedforward module index within layer for 'token_ff' hook target."
        ),
    )

    # Primary Model Params
    ap.add_argument(
        "--primary-embedding-dim",
        type=int,
        default=32,
        help="Embedding dim of the primary model.",
    )
    ap.add_argument(
        "--primary-hidden-dim",
        type=int,
        default=512,
        help="Hidden dim of the primary model (determines SAE input dim).",
    )
    ap.add_argument(
        "--primary-num-layers",
        type=int,
        default=3,
        help="Num layers of the primary model.",
    )
    ap.add_argument(
        "--primary-num-heads",
        type=int,
        default=8,
        help="Num heads of the primary model.",
    )
    ap.add_argument(
        "--primary-num-inducing",
        type=int,
        default=32,
        help="Num inducing points of the primary model.",
    )
    ap.add_argument(
        "--primary-use-layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether primary model uses LayerNorm.",
    )
    ap.add_argument(
        "--primary-dropout",
        type=float,
        default=0.0,
        help="Dropout for primary model (set to 0 for eval).",
    )

    # SAE Hyperparameters
    ap.add_argument(
        "--expansion-factor",
        type=float,
        default=4.0,
        help="SAE expansion factor (hidden_dim / input_dim).",
    )
    ap.add_argument(
        "--l1-coeff", type=float, default=5e-6, help="L1 sparsity coefficient."
    )
    ap.add_argument(
        "--target-usage",
        type=float,
        default=0.05,
        help="Target neuron usage for KL divergence term.",
    )
    ap.add_argument(
        "--usage-coeff",
        type=float,
        default=1.5,
        help=(
            "Base coefficient for the usage KL divergence term (used in "
            "schedule)."
        ),
    )
    ap.add_argument(
        "--dead-neuron-threshold",
        type=float,
        default=1e-8,
        help="Usage threshold below which neurons are considered dead.",
    )
    ap.add_argument(
        "--kl-warmup-steps",
        type=int,
        default=6000,
        help="Number of steps to warm up the KL divergence coefficient.",
    )
    ap.add_argument(
        "--kl-period-steps",
        type=int,
        default=60000,
        help="Number of steps to oscillate the KL divergence coefficient.",
    )
    ap.add_argument(
        "--kl-floor",
        type=float,
        default=0.05,
        help="Floor for the KL divergence coefficient.",
    )

    # Training parameters
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device for training (cuda/cpu).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Number of epochs to train the SAE (outer loop over primary data)."
        ),
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the SAE optimizer.",
    )
    ap.add_argument(
        "--primary-batch-size",
        type=int,
        default=32,
        help="Batch size for the primary model dataloader.",
    )
    ap.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="Size of the activation buffer.",
    )
    ap.add_argument(
        "--sae-batch-size",
        type=int,
        default=1024,
        help="Batch size for SAE training steps.",
    )
    ap.add_argument(
        "--steps-before-train",
        type=int,
        default=50_000,
        help="Number of activations to buffer before starting SAE training.",
    )
    ap.add_argument(
        "--sae-train-steps",
        type=int,
        default=4,
        help=(
            "Number of SAE training steps per primary model step "
            "(after buffer fill)."
        ),
    )
    ap.add_argument(
        "--primary-data-fraction",
        type=float,
        default=0.005,
        help="Fraction of primary data to use for training splits.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() // 2 if os.cpu_count() else 4),
        help="Number of dataloader workers.",
    )
    ap.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )

    # Resampling parameters
    ap.add_argument(
        "--resample-steps",
        type=int,
        nargs="+",
        default=[7000, 14000, 28000, 42000, 56000, 70000],
        help="SAE steps at which to perform dead neuron resampling.",
    )
    ap.add_argument(
        "--resample-weight",
        type=float,
        default=0.2,
        help="Weight factor for resampling encoder weights.",
    )
    ap.add_argument(
        "--resample-bias",
        type=float,
        default=0.0,
        help="Bias value for resampled neurons.",
    )

    # Other
    ap.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose logging during training.",
    )

    args = ap.parse_args()

    # --- Start of main logic using args ---

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    log_file = Path(args.save_dir) / "sae_training_run.log"
    setup_logging(log_file=str(log_file), level=log_level)
    logger = logging.getLogger(__name__)

    # Log all parameters
    logger.info("Starting SAE training CLI script...")
    logger.info("-------------------- Configuration --------------------")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("-------------------------------------------------------")

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Vocabs ---
    try:
        logger.info("Loading vocabs...")
        vocab = load_json_from_path(args.vocab_path)
        weapon_vocab = load_json_from_path(args.weapon_vocab_path)
        pad_token_id = vocab.get("<PAD>")
        if pad_token_id is None:
            raise ValueError("'<PAD>' token not found in vocabulary!")
        logger.info(
            f"Vocab size: {len(vocab)}, Weapon vocab size: {len(weapon_vocab)}"
        )
    except Exception as e:
        logger.exception(f"Failed to load vocabs: {e}")
        sys.exit(1)

    # --- Load Primary Model ---
    try:
        logger.info("Instantiating primary model...")
        primary_model_config = {
            "vocab_size": len(vocab),
            "weapon_vocab_size": len(weapon_vocab),
            "embedding_dim": args.primary_embedding_dim,
            "hidden_dim": args.primary_hidden_dim,
            "output_dim": len(vocab),
            "num_layers": args.primary_num_layers,
            "num_heads": args.primary_num_heads,
            "num_inducing_points": args.primary_num_inducing,
            "use_layer_norm": args.primary_use_layernorm,
            "dropout": args.primary_dropout,
            "pad_token_id": pad_token_id,
        }
        primary_model = SetCompletionModel(**primary_model_config).to(device)

        logger.info(
            f"Loading primary model state dict from: {args.model_checkpoint}"
        )
        if args.model_checkpoint.startswith(("http://", "https://")):
            state_dict = torch.hub.load_state_dict_from_url(
                args.model_checkpoint, map_location=device
            )
        else:
            state_dict = torch.load(args.model_checkpoint, map_location=device)
        primary_model.load_state_dict(state_dict)

        primary_model.eval()
        logger.info("Primary model loaded and set to eval mode.")
    except Exception as e:
        logger.exception(f"Failed to load primary model: {e}")
        sys.exit(1)

    # --- Setup Hook ---
    try:
        logger.info(f"Setting up hook with target: {args.hook_target}")
        hook, handle = setup_hook(
            primary_model,
            target=args.hook_target,
            layer_index=args.hook_layer_index,
            feedforward_module_index=args.hook_ff_module_index,
        )
    except Exception as e:
        logger.exception(f"Failed to setup hook: {e}")
        sys.exit(1)

    # --- Prepare Dataloaders ---
    try:
        logger.info("Loading data and creating dataloaders...")
        df = load_tokenized_data(args.data_csv)
        train_df, val_df, test_df = generate_tokenized_datasets(
            df, frac=args.primary_data_fraction
        )
        logger.info(
            f"Dataset sizes: Train={len(train_df)}, Val={len(val_df)}, "
            f"Test={len(test_df)}"
        )

        train_loader, val_loader, test_loader = generate_dataloaders(
            train_df,
            val_df,
            test_df,
            vocab_size=len(vocab),
            pad_token_id=pad_token_id,
            batch_size=args.primary_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        logger.info("Dataloaders ready.")
    except Exception as e:
        logger.exception(f"Failed to create dataloaders: {e}")
        if "handle" in locals():
            handle.remove()
        sys.exit(1)

    # --- Setup SAE ---
    sae_input_dim = args.primary_hidden_dim
    logger.info(
        "Setting SAE input dimension based on primary model hidden dim: "
        f"{sae_input_dim}"
    )

    sae_config = SAEConfig(
        input_dim=sae_input_dim,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coeff,
        learning_rate=args.lr,
        resample_interval=999_999_999,
        dead_neuron_threshold=args.dead_neuron_threshold,
        target_usage=args.target_usage,
        usage_coeff=args.usage_coeff,
        dead_neuron_steps=1,
    )

    sae_model = SparseAutoencoder(
        input_dim=sae_input_dim,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coeff,
        target_usage=args.target_usage,
        usage_coeff=args.usage_coeff,
        dead_neuron_threshold=args.dead_neuron_threshold,
    ).to(device)

    logger.info(
        "SAE instantiated: Input Dim=%d, Hidden Dim=%d",
        sae_input_dim,
        sae_model.hidden_dim,
    )

    optimizer = AdamW(sae_model.parameters(), lr=args.lr)

    # Estimate total SAE steps for scheduler
    try:
        estimated_batches_per_epoch = len(train_loader)
        total_primary_steps = args.epochs * estimated_batches_per_epoch
        primary_steps_for_buffer = (
            args.steps_before_train
            // (args.primary_batch_size * args.sae_train_steps)
            if args.primary_batch_size > 0
            else 0
        )
        primary_steps_with_sae = max(
            0, total_primary_steps - primary_steps_for_buffer
        )
        total_sae_steps = primary_steps_with_sae * args.sae_train_steps
        if total_sae_steps <= 0:
            logger.warning(
                "Estimated total SAE steps is non-positive. Scheduler T_max "
                "might be too small. Falling back to primary steps."
            )
            total_sae_steps = total_primary_steps
        logger.info(
            "Estimated total SAE training steps for scheduler: %d",
            total_sae_steps,
        )
    except Exception as e:
        logger.error(
            "Could not estimate total SAE steps: %s. Using 100k as fallback.",
            e,
        )
        total_sae_steps = 100_000

    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_sae_steps, eta_min=args.lr * 0.1
    )

    # --- Save Run Config ---
    config_save_path = save_dir / "sae_run_config.json"
    run_config = vars(args)
    run_config["sae_input_dim"] = sae_input_dim
    run_config["sae_hidden_dim"] = sae_model.hidden_dim
    for k, v in run_config.items():
        if isinstance(v, Path):
            run_config[k] = str(v)

    logger.info(f"Saving run configuration to {config_save_path}")
    try:
        with open(config_save_path, "wb") as f:
            f.write(
                orjson.dumps(
                    run_config,
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                )
            )
    except Exception as e:
        logger.error(f"Failed to save run config: {e}")

    # --- Start Training ---
    logger.info("Starting SAE training loop...")
    metrics_history = None
    final_sae_model_state = None
    training_successful = False
    try:
        metrics_history = train_sae_model(
            primary_model=primary_model,
            sae_model=sae_model,
            optimizer=optimizer,
            scheduler=scheduler,
            hook=hook,
            sae_config=sae_config,
            data_loader=train_loader,
            val_loader=val_loader,
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
        final_sae_model_state = sae_model.state_dict()
        training_successful = True
        logger.info("SAE training loop completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during SAE training: {e}")
        sae_save_path = save_dir / "sae_model_FAILED.pth"
        try:
            torch.save(sae_model.state_dict(), sae_save_path)
            logger.info(
                f"Saved failed/partial SAE model state to {sae_save_path}"
            )
        except Exception as save_err:
            logger.error(f"Could not save failed model state: {save_err}")
    finally:
        handle.remove()
        logger.info("Training hook removed.")

    # --- Save Final Model & Training Metrics ---
    if training_successful and final_sae_model_state is not None:
        sae_save_path = save_dir / "sae_model_final.pth"
        torch.save(final_sae_model_state, sae_save_path)
        logger.info(f"Saved Final SAE model to {sae_save_path}")

        if metrics_history:
            metrics_save_path = save_dir / "sae_metrics_history.json"
            try:
                with open(metrics_save_path, "wb") as f:
                    f.write(
                        orjson.dumps(
                            metrics_history,
                            option=orjson.OPT_INDENT_2
                            | orjson.OPT_SERIALIZE_NUMPY,
                        )
                    )
                logger.info(f"Saved metrics history to {metrics_save_path}")
            except Exception as e:
                logger.error(f"Failed to save metrics history: {e}")

    # --- Final Testing Phase ---
    if training_successful and final_sae_model_state is not None:
        logger.info("Starting final evaluation on the test set...")
        sae_model.load_state_dict(final_sae_model_state)
        sae_model.to(device).eval()
        primary_model.to(device).eval()

        test_handle = None
        try:
            test_hook, test_handle = setup_hook(
                primary_model,
                target=args.hook_target,
                layer_index=args.hook_layer_index,
                feedforward_module_index=args.hook_ff_module_index,
            )

            test_metrics = evaluate_sae_model(
                primary_model=primary_model,
                sae_model=sae_model,
                hook=test_hook,
                data_loader=test_loader,
                device=device,
                sae_config=sae_config,
                vocab=vocab,
                description="Testing",
            )

            logger.info("--- Test Set Evaluation Results ---")
            for key, value in test_metrics.items():
                log_value = (
                    float(value)
                    if isinstance(value, (torch.Tensor, np.number, float, int))
                    else value
                )
                logger.info(
                    f"  {key}: {log_value:.6f}"
                    if isinstance(log_value, float)
                    else f"  {key}: {log_value}"
                )

            test_metrics_save_path = save_dir / "sae_test_metrics.json"
            try:
                with open(test_metrics_save_path, "wb") as f:
                    f.write(
                        orjson.dumps(
                            test_metrics,
                            option=orjson.OPT_INDENT_2
                            | orjson.OPT_SERIALIZE_NUMPY
                            | orjson.OPT_SORT_KEYS,
                        )
                    )
                logger.info(f"Saved test metrics to {test_metrics_save_path}")
            except Exception as e:
                logger.error(f"Failed to save test metrics: {e}")

        except Exception as e:
            logger.exception(f"An error occurred during final testing: {e}")
        finally:
            if test_handle is not None:
                test_handle.remove()
                logger.info("Test hook removed.")
    else:
        logger.warning(
            "Skipping final testing because training did not complete "
            "successfully or model state is unavailable."
        )

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
