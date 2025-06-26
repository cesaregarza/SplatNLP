import logging

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.data_objects import (
    ActivationBuffer,
    ActivationHook,
    SAEConfig,
)
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.monosemantic_sae.sae_training.evaluate import (
    evaluate_reconstruction_impact,
    evaluate_sae_model,
)
from splatnlp.monosemantic_sae.sae_training.resample import (
    resample_dead_neurons,
)
from splatnlp.monosemantic_sae.sae_training.schedules import (
    usage_coeff_schedule,
)

logger = logging.getLogger(__name__)


def train_sae_model(
    primary_model: SetCompletionModel,
    sae_model: SparseAutoencoder,
    optimizer: AdamW,
    scheduler: _LRScheduler,
    hook: ActivationHook,
    sae_config: SAEConfig,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    vocab: dict[str, int],
    device: torch.device,
    num_epochs: int,
    activation_buffer_size: int,
    sae_batch_size: int,
    steps_before_sae_train: int,
    sae_train_steps_per_primary_step: int,
    resample_steps: set[int],
    resample_weight: float,
    resample_bias: float,
    kl_warmup_steps: int = 6000,
    kl_period_steps: int = 60000,
    kl_floor: float = 0.05,
    log_interval: int = 500,
    gradient_clip_val: float = 1.0,
    verbose: bool = True,
) -> list[dict[str, float | int | bool]]:
    """
    Main SAE training loop function. Performs:
    - Activation buffering from primary model forward passes
    - SAE training steps (with KL warm-up scheduling)
    - Periodic dead-neuron resampling
    - Validation (both standard SAE MSE/sparsity and reconstruction impact)

    Args:
        primary_model: The primary model that generates activations, a
            SetCompletionModel instance
        sae_model: The sparse autoencoder model to train
        optimizer: Optimizer for SAE training
        scheduler: Learning rate scheduler
        hook: Hook for capturing activations from primary model
        sae_config: Configuration for the SAE
        data_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        vocab: Vocabulary dictionary
        device: Device to perform computations on
        num_epochs: Number of training epochs
        activation_buffer_size: Size of activation buffer
        sae_batch_size: Batch size for SAE training
        steps_before_sae_train: Number of steps to buffer before starting SAE
            training
        sae_train_steps_per_primary_step: Number of SAE training steps per
            primary model step
        resample_steps: Set of steps at which to resample dead neurons
        resample_weight: Weight to apply to normalized resampled inputs
        resample_bias: Bias to apply to resampled neurons
        kl_warmup_steps: Number of steps for KL coefficient warmup
        kl_period_steps: Period of cosine oscillation after warmup
        kl_floor: Minimum KL coefficient value
        log_interval: Interval for logging metrics
        gradient_clip_val: Value for gradient clipping
        verbose: Whether to show progress bars

    Returns:
        list[dict]: List of metric dictionaries collected during training
    """
    global_step = 0
    sae_step = 0
    metrics_history: list[dict[str, float | int | bool]] = []

    # Activation Buffer
    act_buf = ActivationBuffer(
        max_size=activation_buffer_size, d=sae_config.input_dim, device=device
    )

    pad_id = vocab.get("<PAD>", -1)

    logger.info("Activation buffer size: %d", act_buf.max_size)
    logger.info("SAE input dimension: %d", sae_config.input_dim)
    logger.info("KL warm-up steps: %d", kl_warmup_steps)
    logger.info(f"Gradient clipping value: {gradient_clip_val}")

    for epoch in range(num_epochs):
        logger.info(
            "--- Starting Primary Model Epoch %d/%d ---", epoch + 1, num_epochs
        )

        # Keep primary model in eval mode
        primary_model.eval()
        # Set SAE to train mode
        sae_model.train()

        pbar_train: tqdm[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = tqdm(
            data_loader,
            disable=not verbose,
            desc=f"Epoch {epoch+1} Training",
            leave=False,
            position=0,
        )

        for abilities, weapons, targets, _ in pbar_train:
            global_step += 1

            abilities = abilities.to(device)
            weapons = weapons.to(device)
            targets = targets.to(device)
            key_padding_mask = (abilities == pad_id).to(device)

            # 1) Collect activations via hook
            hook.clear_activations()
            with torch.no_grad():
                _ = primary_model(
                    abilities, weapons, key_padding_mask=key_padding_mask
                )
            acts = hook.get_and_clear()

            # Basic validation of acts
            if (
                acts is None
                or acts.dim() != 2
                or acts.shape[1] != sae_config.input_dim
                or acts.shape[0] == 0
            ):
                continue
            act_buf.add(acts.detach())

            # 2) Only start SAE training if buffer has enough samples
            if len(act_buf) < steps_before_sae_train:
                pbar_train.set_description(
                    f"Epoch {epoch+1} Buffering "
                    f"{len(act_buf)}/{steps_before_sae_train}"
                )
                continue

            # Train SAE
            if not sae_model.training:
                sae_model.train()

            pbar_train.set_description(f"Epoch {epoch+1} SAE-Step {sae_step}")

            buf_loader = act_buf.get_loader(
                batch_size=sae_batch_size, shuffle=True
            )
            buf_iter = iter(buf_loader)

            for _ in range(sae_train_steps_per_primary_step):
                try:
                    (batch,) = next(buf_iter)
                except StopIteration:
                    buf_iter = iter(
                        act_buf.get_loader(
                            batch_size=sae_batch_size, shuffle=True
                        )
                    )
                    batch: torch.Tensor = next(buf_iter)[0]

                # --- KL warm-up / usage schedule ---
                desired_coeff = usage_coeff_schedule(
                    sae_step,
                    base=sae_config.usage_coeff,
                    warmup_steps=kl_warmup_steps,
                    period_steps=kl_period_steps,
                    floor=kl_floor,
                )
                # If `set_kl_coeff` method is defined, use it; otherwise set \
                # usage_coeff directly
                if hasattr(sae_model, "set_kl_coeff"):
                    sae_model.set_kl_coeff(desired_coeff)
                else:
                    sae_model.usage_coeff = desired_coeff

                current_actual_coeff = getattr(
                    sae_model, "usage_coeff", desired_coeff
                )

                # --- SAE Training Step ---
                metrics = sae_model.training_step(
                    batch.float(), optimizer, gradient_clip_val
                )
                sae_step += 1
                scheduler.step()

                # Collect extra stats
                with torch.no_grad():
                    hidden: torch.Tensor = sae_model(batch.float())[1]
                    sparsity = (hidden.abs() > 1e-6).float().mean().item()
                    metrics["sparsity_l0"] = sparsity

                metrics["sae_step"] = sae_step
                metrics["global_step"] = global_step
                metrics["kl_coeff_target"] = desired_coeff
                metrics["kl_coeff_actual"] = current_actual_coeff
                metrics["epoch"] = epoch + 1
                metrics["lr"] = scheduler.get_last_lr()[0]
                metrics["buffer_fill"] = len(act_buf) / act_buf.max_size

                metrics_history.append(metrics.copy())

                if sae_step % log_interval == 0:
                    log_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k
                        not in [
                            "sae_step",
                            "global_step",
                            "epoch",
                            "buffer_fill",
                        ]
                    }
                    m_str = ", ".join(
                        [f"{k}: {v:.4g}" for k, v in log_metrics.items()]
                    )
                    # Clear the progress bar before logging to prevent overlap
                    pbar_train.clear()
                    logger.info(
                        "Epoch %d Step %d LR %.2e KL %.3f | %s",
                        epoch + 1,
                        sae_step,
                        metrics["lr"],
                        current_actual_coeff,
                        m_str,
                    )

                # 3) Resample dead neurons if needed
                if sae_step in resample_steps:
                    logger.info("-- Resampling at SAE step %d --", sae_step)
                    sae_model.eval()
                    n_resampled = resample_dead_neurons(
                        sae_model,
                        act_buf,
                        optimizer,
                        sae_config,
                        device,
                        resample_weight=resample_weight,
                        resample_bias=resample_bias,
                    )
                    sae_model.train()
                    logger.info("-- Resampled %d neurons --", n_resampled)
                    metrics_history[-1]["resampled_neurons"] = n_resampled
                    metrics_history[-1]["is_resample_step"] = True

        # End of epoch
        logger.info(
            "--- Finished Epoch %d (SAE Step %d) ---",
            epoch + 1,
            sae_step,
        )

        # Validation
        logger.info("--- Starting Validation for Epoch %d ---", epoch + 1)
        primary_model.eval()
        sae_model.eval()

        # 1) Standard SAE MSE/sparsity metrics
        try:
            standard_val_metrics = evaluate_sae_model(
                primary_model=primary_model,
                sae_model=sae_model,
                hook=hook,
                data_loader=val_loader,
                device=device,
                sae_config=sae_config,
                vocab=vocab,
                description=f"Epoch {epoch+1} Std Val",
            )
            logger.info(
                "--- Standard Validation Results Epoch %d ---",
                epoch + 1,
            )
            std_val_log_str = ", ".join(
                [
                    f"{k}: {v:.5f}"
                    for k, v in standard_val_metrics.items()
                    if isinstance(v, (float, int))
                ]
            )
            logger.info(std_val_log_str)

            standard_val_metrics["epoch"] = epoch + 1
            standard_val_metrics["sae_step"] = sae_step
            standard_val_metrics["global_step"] = global_step
            standard_val_metrics["is_validation_standard"] = True
            metrics_history.append(standard_val_metrics)
        except Exception as std_val_err:
            logger.error(
                "Error during standard validation: %s",
                std_val_err,
                exc_info=True,
            )

        # 2) Reconstruction impact validation
        try:
            impact_val_metrics = evaluate_reconstruction_impact(
                primary_model=primary_model,
                sae_model=sae_model,
                hook=hook,
                data_loader=val_loader,
                device=device,
                sae_config=sae_config,
                vocab=vocab,
                description=f"Epoch {epoch+1} Impact Val",
            )
            logger.info(
                "--- Reconstruction Impact Validation Results Epoch %d ---",
                epoch + 1,
            )
            impact_val_log_str: str = ", ".join(
                [
                    f"{k}: {v:.5f}"
                    for k, v in impact_val_metrics.items()
                    if isinstance(v, (float, int))
                ]
            )
            logger.info(impact_val_log_str)

            impact_val_metrics["epoch"] = epoch + 1
            impact_val_metrics["sae_step"] = sae_step
            impact_val_metrics["global_step"] = global_step
            impact_val_metrics["is_validation_impact"] = True
            metrics_history.append(impact_val_metrics)
        except Exception as impact_val_err:
            logger.error(
                "Error during reconstruction impact validation: %s",
                impact_val_err,
                exc_info=True,
            )

        # Log usage stats
        usage = sae_model.usage_ema.cpu().numpy()
        dead_thresh = sae_config.dead_neuron_threshold
        dead = (usage < dead_thresh).sum()
        active = len(usage) - dead

        epoch_summary: dict[str, float | int | bool] = {
            "epoch": epoch + 1,
            "sae_step": sae_step,
            "global_step": global_step,
            "dead_neurons": int(dead),
            "active_neurons": int(active),
            "dead_percent": (
                float(100 * dead / len(usage)) if len(usage) > 0 else 0
            ),
            "mean_usage": float(usage.mean()) if len(usage) > 0 else 0,
            "std_usage": float(usage.std()) if len(usage) > 0 else 0,
            "min_usage": float(usage.min()) if len(usage) > 0 else 0,
            "max_usage": float(usage.max()) if len(usage) > 0 else 0,
            "is_epoch_summary": True,
        }

        if len(usage) > 0:
            prc = torch.tensor(usage).quantile(
                torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            )
            epoch_summary.update(
                {
                    "p10_usage": float(prc[0]),
                    "p25_usage": float(prc[1]),
                    "p50_usage": float(prc[2]),
                    "p75_usage": float(prc[3]),
                    "p90_usage": float(prc[4]),
                    "p95_usage": float(prc[5]),
                    "p99_usage": float(prc[6]),
                }
            )

        metrics_history.append(epoch_summary)
        logger.info(
            (
                "Usage Stats: Dead=%d (%.2f%%) "
                "| Mean=%.4g "
                "| Std=%.4g "
                "| Median=%.4g "
                "| Max=%.4g",
            ),
            dead,
            epoch_summary["dead_percent"],
            epoch_summary["mean_usage"],
            epoch_summary["std_usage"],
            epoch_summary.get("p50_usage", 0),
            epoch_summary["max_usage"],
        )

        # End of epoch - set SAE back to train mode for next epoch
        sae_model.train()

    logger.info("\n--- SAE Training Finished ---\n")
    return metrics_history
