"""
Updated SAE training script with KL-warm-up, sharper sparsity defaults and
clean hooks so you can hit the 1e-3 usage band without touching the rest of
your pipeline.

Key additions
-------------
1. **KL warm-up** - linear ramp of the usage (KL) coefficient from 0 -> 1 over
   `kl_warmup_steps` SAE optimisation steps.
2. **Stronger dead-neuron test** - default `dead_neuron_threshold=1e-3`.
3. **Convenience helper** in `SparseAutoencoder` expected interface:
   `set_kl_coeff(float)` so the training loop can change the coefficient on
   the fly without rebuilding the optimiser.
4. **Extra resample points** (optional):   7k, 14k, 28k, 42k, 56k, 70k.

Drop-in compatible with the previous script - if you haven't updated the
`SparseAutoencoder` yet, add the 4-line method shown in the comments below.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# Local imports
from .data_objects import ActivationBuffer, ActivationHook, SAEConfig
from .models import SparseAutoencoder

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_fh = logging.FileHandler(f"logs/sae_training_{_ts}.log")
_fh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(_fh)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dead-neuron resampling -----------------------------------------------------
# ---------------------------------------------------------------------------
@torch.no_grad()
def resample_dead_neurons(
    sae_model: SparseAutoencoder,
    activation_buffer: ActivationBuffer,
    optimizer: Optimizer,
    sae_config: SAEConfig,
    device: torch.device,
    resample_weight: float = 0.2,
    resample_bias: float = 0.0,
) -> int:
    """Same as before but uses *sae_config.dead_neuron_threshold* (now 1e-3)."""
    dead_neurons = sae_model.get_dead_neurons(
        threshold=sae_config.dead_neuron_threshold
    )
    n_dead = len(dead_neurons)
    if n_dead == 0:
        return 0

    logger.info(f"Resampling {n_dead} dead neurons")
    sample_size = min(len(activation_buffer), n_dead * 10)
    if sample_size == 0:
        logger.warning("Activation buffer empty - skip resampling")
        return 0

    buffer_loader = activation_buffer.get_loader(
        batch_size=sample_size, shuffle=False
    )
    (inputs_sample,) = next(iter(buffer_loader))

    recon, _ = sae_model(inputs_sample)
    mse = F.mse_loss(
        recon.float(), inputs_sample.float(), reduction="none"
    ).mean(dim=1)

    if torch.isnan(mse).any() or mse.sum() < 1e-12:
        idx = torch.randint(0, inputs_sample.size(0), (n_dead,), device=device)
    else:
        probs = (mse / mse.sum()).cpu()
        idx = torch.multinomial(probs, n_dead, replacement=True).to(device)

    resample_inputs = inputs_sample[idx]
    normed = F.normalize(resample_inputs, dim=1)

    sae_model.encoder.weight.data[dead_neurons] = normed * resample_weight
    sae_model.encoder.bias.data[dead_neurons] = resample_bias
    sae_model.decoder.weight.data[:, dead_neurons] = (
        sae_model.encoder.weight.data[dead_neurons].T
    )

    # Zero Adam moments so the fresh weights aren't dragged back to 0
    if isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if not state:  # param not in state dict yet
                    continue
                if p is sae_model.encoder.weight or p is sae_model.encoder.bias:
                    state["exp_avg"][dead_neurons] = 0
                    state["exp_avg_sq"][dead_neurons] = 0
                elif p is sae_model.decoder.weight:
                    state["exp_avg"][:, dead_neurons] = 0
                    state["exp_avg_sq"][:, dead_neurons] = 0

    # Reset running usage estimate for those neurons
    live = sae_model.usage_ema > sae_config.dead_neuron_threshold
    sae_model.usage_ema[dead_neurons] = (
        sae_model.usage_ema[live].mean()
        if live.any()
        else sae_config.target_usage
    )

    logger.info("Finished resampling %d neurons", n_dead)
    return n_dead


def usage_coeff_schedule(
    sae_step: int,
    base: float = 1.5,
    warmup_steps: int = 6_000,
    period_steps: int = 60_000,
    floor: float = 0.05,
) -> float:
    if sae_step < warmup_steps:
        return floor + (base - floor) * sae_step / warmup_steps

    # cosine from `base` down to `floor` and back every `period_steps`
    phase = (sae_step - warmup_steps) % period_steps
    cos_term = 0.5 * (1 + np.cos(2 * np.pi * phase / period_steps))
    return floor + (base - floor) * cos_term


# ---------------------------------------------------------------------------
# SAE training loop ---------------------------------------------------------
# ---------------------------------------------------------------------------


def train_sae_model(
    primary_model: nn.Module,
    sae_model: SparseAutoencoder,
    optimizer: AdamW,
    scheduler: _LRScheduler,
    hook: ActivationHook,
    sae_config: SAEConfig,
    data_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    vocab: dict,
    device: torch.device,
    num_epochs: int,
    activation_buffer_size: int,
    sae_batch_size: int,
    steps_before_sae_train: int,
    sae_train_steps_per_primary_step: int,
    resample_steps: set[int],
    resample_weight: float,
    resample_bias: float,
    *,
    kl_warmup_steps: int = 6_000,
    kl_period_steps: int = 60_000,
    kl_floor: float = 0.05,
    log_interval: int = 500,
    gradient_clip_val: float = 1.0,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Training loop with KL-warm-up and validation."""
    global_step = 0
    sae_step = 0
    metrics_history = []
    act_buf = ActivationBuffer(
        max_size=activation_buffer_size, d=sae_config.input_dim, device=device
    )
    pad_id = vocab.get("<PAD>", -1)
    last_log_time = time.time()

    logger.info("Activation buffer size: %d", act_buf.max_size)
    logger.info("SAE input dimension: %d", sae_config.input_dim)
    logger.info("KL warm-up steps: %d", kl_warmup_steps)
    logger.info(f"Gradient clipping value: {gradient_clip_val}")

    for epoch in range(num_epochs):
        logger.info(
            "--- Starting Primary Model Epoch %d/%d ---", epoch + 1, num_epochs
        )
        pbar_train = tqdm(
            data_loader,
            disable=not verbose,
            desc=f"Epoch {epoch+1} Training",
            leave=False,
        )

        primary_model.eval()  # Keep primary model in eval mode
        sae_model.train()  # Set SAE to train mode for the epoch training phase

        # --- Training Phase for the Epoch ---
        for abilities, weapons, targets, _ in pbar_train:
            global_step += 1

            abilities = abilities.to(device)
            weapons = weapons.to(device)
            targets = targets.to(
                device
            )  # Although targets aren't used by SAE itself
            key_padding_mask = (abilities == pad_id).to(device)

            # 1) Collect activations
            hook.clear_activations()
            with (
                torch.no_grad()
            ):  # Primary model forward pass doesn't need gradients here
                _ = primary_model(
                    abilities, weapons, key_padding_mask=key_padding_mask
                )
            acts = hook.get_and_clear()
            if (
                acts is None
                or acts.dim() != 2
                or acts.shape[1] != sae_config.input_dim
                or acts.shape[0] == 0  # Skip empty batches
            ):
                continue
            act_buf.add(acts.detach())

            if len(act_buf) < steps_before_sae_train:
                pbar_train.set_description(
                    f"Epoch {epoch+1} Buffering {len(act_buf)}/{steps_before_sae_train}"
                )
                continue

            # 2) Train SAE
            # Ensure SAE is in training mode before training steps
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
                    (batch,) = next(buf_iter)

                # --- KL warm-up ---
                desired_coeff = usage_coeff_schedule(
                    sae_step,
                    sae_config.usage_coeff,  # Base value from config (might be overwritten by schedule)
                    kl_warmup_steps,
                    kl_period_steps,
                    kl_floor,
                )
                # Ensure coeff doesn't become exactly zero if floor is zero, maybe add small epsilon?
                # desired_coeff = max(desired_coeff, 1e-9) # Optional: prevent exactly zero coeff

                if hasattr(sae_model, "set_kl_coeff"):
                    sae_model.set_kl_coeff(desired_coeff)
                elif hasattr(sae_model, "usage_coeff"):
                    # Fallback: Directly set if method doesn't exist (less ideal)
                    if (
                        epoch == 0 and sae_step < 10
                    ):  # Log only initially if using fallback
                        logger.warning(
                            "SAE model does not have set_kl_coeff method, directly setting usage_coeff attribute."
                        )
                    sae_model.usage_coeff = desired_coeff
                current_actual_coeff = getattr(
                    sae_model, "usage_coeff", desired_coeff
                )  # Get the coeff actually used

                # --- SAE Training Step ---
                # Make sure training_step accepts gradient_clip_val if needed internally
                # Or perform clipping here after loss.backward()
                metrics = sae_model.training_step(
                    batch.float(),
                    optimizer,
                    gradient_clip_val,  # Pass clipping value
                )
                sae_step += 1
                scheduler.step()  # Step the LR scheduler

                # --- Calculate and Log Metrics ---
                with torch.no_grad():
                    _, hidden = sae_model(batch.float())
                    # Use a small threshold for sparsity calculation
                    sparsity_thresh = 1e-6
                    sparsity = (
                        (hidden.abs() > sparsity_thresh).float().mean().item()
                    )
                    metrics["sparsity_l0"] = sparsity  # L0 norm approx
                    # metrics["l1_loss"] is often returned by training_step, rename if needed
                    if "l1_loss" not in metrics and hasattr(
                        sae_model, "l1_coefficient"
                    ):
                        metrics["l1_loss_term"] = (
                            torch.abs(hidden).sum(dim=-1).mean()
                            * sae_model.l1_coefficient
                        ).item()

                metrics["sae_step"] = sae_step
                metrics["global_step"] = global_step
                metrics["kl_coeff_target"] = desired_coeff
                metrics["kl_coeff_actual"] = (
                    current_actual_coeff  # Log the actual coeff
                )
                metrics["epoch"] = epoch + 1
                metrics["lr"] = scheduler.get_last_lr()[0]  # Log learning rate
                metrics["buffer_fill"] = (
                    len(act_buf) / act_buf.max_size
                )  # Log buffer fill rate

                metrics_history.append(metrics.copy())  # Store training metrics

                if sae_step % log_interval == 0:
                    # Format metrics for logging
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
                    logger.info(
                        f"Ep {epoch+1} Step {sae_step} LR {metrics['lr']:.2e} KL {current_actual_coeff:.3f} | {m_str}"
                    )

                # 3) Resample dead neurons
                if sae_step in resample_steps:
                    logger.info("-- Resampling at SAE step %d --", sae_step)
                    # Ensure SAE is in eval mode for resampling stats if needed, then back to train
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
                    sae_model.train()  # Set back to train mode
                    logger.info(f"-- Resampled {n_resampled} neurons --")
                    # Record resampling event in metrics history
                    metrics_history[-1]["resampled_neurons"] = n_resampled
                    # Add a marker for resampling step for easier plotting
                    metrics_history[-1]["is_resample_step"] = True

        # --- End of Epoch ---
        logger.info(f"--- Finished Epoch {epoch + 1} (SAE Step {sae_step}) ---")

        # --- Validation Phase ---
        logger.info(f"--- Starting Validation for Epoch {epoch + 1} ---")
        primary_model.eval()  # Ensure both models are in eval mode
        sae_model.eval()

        # --- Standard Validation (SAE MSE, Sparsity, etc.) ---
        try:
            logger.info("Running standard SAE validation...")
            standard_val_metrics = evaluate_sae_model(  # Original eval function
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
                f"--- Standard Validation Results Epoch {epoch + 1} ---"
            )
            std_val_log_str = ", ".join(
                [
                    f"{k}: {v:.5f}"
                    for k, v in standard_val_metrics.items()
                    if isinstance(v, (float, int))
                ]
            )
            logger.info(std_val_log_str)

            # Add standard validation metrics to history
            standard_val_metrics["epoch"] = epoch + 1
            standard_val_metrics["sae_step"] = sae_step
            standard_val_metrics["global_step"] = global_step
            standard_val_metrics["is_validation_standard"] = True
            metrics_history.append(standard_val_metrics)

        except Exception as std_val_err:
            logger.error(
                f"Error during standard validation: {std_val_err}",
                exc_info=True,
            )

        # --- Reconstruction Impact Validation (Logits, Loss, Acc) ---
        try:
            logger.info("Running reconstruction impact validation...")
            impact_val_metrics = (
                evaluate_reconstruction_impact(  # New eval function
                    primary_model=primary_model,
                    sae_model=sae_model,
                    hook=hook,
                    data_loader=val_loader,  # Use the same validation loader
                    device=device,
                    sae_config=sae_config,
                    vocab=vocab,
                    description=f"Epoch {epoch+1} Impact Val",
                )
            )
            logger.info(
                f"--- Reconstruction Impact Validation Results Epoch {epoch + 1} ---"
            )
            impact_val_log_str = ", ".join(
                [
                    f"{k}: {v:.5f}"
                    for k, v in impact_val_metrics.items()
                    if isinstance(v, (float, int))
                ]
            )
            logger.info(impact_val_log_str)

            # Add impact validation metrics to history
            impact_val_metrics["epoch"] = epoch + 1
            impact_val_metrics["sae_step"] = sae_step
            impact_val_metrics["global_step"] = global_step
            impact_val_metrics["is_validation_impact"] = True
            metrics_history.append(impact_val_metrics)

        except Exception as impact_val_err:
            logger.error(
                f"Error during reconstruction impact validation: {impact_val_err}",
                exc_info=True,
            )

        # --- Log Neuron Usage Stats (as before) ---
        # Ensure model is in eval mode to get consistent usage stats if needed
        sae_model.eval()
        usage = sae_model.usage_ema.cpu().numpy()
        dead_thresh = sae_config.dead_neuron_threshold
        dead = (usage < dead_thresh).sum()
        active = len(usage) - dead

        epoch_summary = {
            "epoch": epoch + 1,
            "sae_step": sae_step,
            "global_step": global_step,
            "dead_neurons": int(dead),
            "active_neurons": int(active),
            "dead_percent": (
                float(100 * dead / len(usage)) if len(usage) > 0 else 0
            ),
            # ... (other usage stats: mean, std, percentiles) ...
            "mean_usage": float(np.mean(usage)) if len(usage) > 0 else 0,
            "std_usage": float(np.std(usage)) if len(usage) > 0 else 0,
            "min_usage": float(np.min(usage)) if len(usage) > 0 else 0,
            "max_usage": float(np.max(usage)) if len(usage) > 0 else 0,
        }
        if len(usage) > 0:
            prc = np.percentile(usage, [10, 25, 50, 75, 90, 95, 99])
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
        epoch_summary["is_epoch_summary"] = True
        metrics_history.append(epoch_summary)  # Append summary stats

        usage_log_str = (
            f"Usage Stats: Dead={dead} ({epoch_summary['dead_percent']:.2f}%) | "
            f"Mean={epoch_summary['mean_usage']:.4g} | Std={epoch_summary['std_usage']:.4g} | "
            f"Median={epoch_summary.get('p50_usage', 0):.4g} | Max={epoch_summary['max_usage']:.4g}"
        )
        logger.info(usage_log_str)

        # Set model back to train mode for the next epoch's training phase
        sae_model.train()

    logger.info("\n--- SAE Training Finished ---\n")
    return metrics_history


@torch.no_grad()  # Ensure no gradients are computed
def evaluate_sae_model(
    primary_model: nn.Module,
    sae_model: SparseAutoencoder,
    hook: ActivationHook,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sae_config: SAEConfig,
    vocab: dict,
    description: str = "Validation",  # Description for progress bar
) -> dict[str, float]:
    """Evaluates the SAE model on a given dataset."""
    primary_model.eval()
    sae_model.eval()

    total_mse_loss = 0.0
    total_l1_loss = 0.0
    total_sparsity = 0.0  # L0 norm (fraction of non-zero activations)
    total_feature_magnitude = 0.0  # Average magnitude of hidden features
    num_batches = 0
    num_activations = 0

    pad_id = vocab.get("<PAD>", -1)

    pbar_eval = tqdm(
        data_loader, desc=description, leave=False
    )  # Use leave=False for nested loops

    for abilities, weapons, targets, _ in pbar_eval:
        abilities = abilities.to(device)
        weapons = weapons.to(device)
        # targets = targets.to(device) # Targets not needed for SAE evaluation
        key_padding_mask = (abilities == pad_id).to(device)

        # 1) Collect activations
        hook.clear_activations()
        _ = primary_model(abilities, weapons, key_padding_mask=key_padding_mask)
        acts = hook.get_and_clear()

        if (
            acts is None
            or acts.dim() != 2
            or acts.shape[1] != sae_config.input_dim
        ):
            logger.warning(
                f"Skipping batch in {description} due to invalid activations shape: {acts.shape if acts is not None else 'None'}"
            )
            continue
        if acts.shape[0] == 0:  # Skip empty activation batches
            continue

        current_batch_size = acts.shape[
            0
        ]  # Number of activation vectors in this batch
        num_activations += current_batch_size

        # 2) Run SAE forward pass
        recon_acts, hidden_acts = sae_model(
            acts.float()
        )  # Ensure input is float

        # 3) Calculate losses and metrics for this batch
        batch_mse_loss = F.mse_loss(recon_acts, acts.float()).item()
        batch_l1_loss = (
            torch.norm(hidden_acts, p=1, dim=-1).mean().item()
        )  # Avg L1 norm per activation vector
        batch_sparsity = (
            (hidden_acts > 1e-6).float().mean().item()
        )  # Fraction active (> threshold)
        batch_feature_mag = (
            torch.abs(hidden_acts).mean().item()
        )  # Average absolute feature value

        # Accumulate weighted losses (by number of activations in batch)
        total_mse_loss += batch_mse_loss * current_batch_size
        total_l1_loss += batch_l1_loss * current_batch_size
        total_sparsity += batch_sparsity * current_batch_size
        total_feature_magnitude += batch_feature_mag * current_batch_size
        num_batches += 1

        pbar_eval.set_postfix(
            mse=batch_mse_loss, l1=batch_l1_loss, sparsity=batch_sparsity
        )

    if num_activations == 0:
        logger.warning(
            f"{description} loop completed without processing any activations."
        )
        return {
            f"{description.lower()}_mse_loss": 0.0,
            f"{description.lower()}_l1_loss": 0.0,
            f"{description.lower()}_sparsity": 0.0,
            f"{description.lower()}_feature_magnitude": 0.0,
            "num_eval_batches": num_batches,
            "num_eval_activations": num_activations,
        }

    # Calculate average metrics over the entire dataset
    avg_mse_loss = total_mse_loss / num_activations
    avg_l1_loss = total_l1_loss / num_activations
    avg_sparsity = total_sparsity / num_activations
    avg_feature_magnitude = total_feature_magnitude / num_activations

    return {
        f"{description.lower()}_mse_loss": avg_mse_loss,
        f"{description.lower()}_l1_loss": avg_l1_loss,
        f"{description.lower()}_sparsity": avg_sparsity,
        f"{description.lower()}_feature_magnitude": avg_feature_magnitude,
        "num_eval_batches": num_batches,
        "num_eval_activations": num_activations,
    }


@torch.no_grad()  # Ensure no gradients are computed during evaluation
def evaluate_reconstruction_impact(
    primary_model: nn.Module,
    sae_model: nn.Module,
    hook: ActivationHook,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sae_config: SAEConfig,
    vocab: dict,
    description: str = "Reconstruction Impact Eval",
) -> dict[str, float]:
    """
    Evaluates the impact of SAE reconstruction on the primary model's output logits.

    Assumes the hook captures activations directly preceding primary_model.output_layer.
    Adapts to handle potentially multi-dimensional targets by comparing against the first target token.
    """
    primary_model.eval()
    sae_model.eval()

    total_logit_mse = 0.0
    total_logit_cosine_sim = 0.0
    total_kl_div = 0.0
    total_orig_loss = 0.0
    total_recon_loss = 0.0
    total_top1_acc_orig = 0.0
    total_top1_acc_recon = 0.0
    total_top1_match = 0.0  # Does top prediction match between orig and recon?
    valid_samples_for_acc_loss = (
        0  # Count samples where acc/loss could be computed
    )

    num_batches = 0
    num_samples = 0  # Count total number of sequences processed

    pad_id = vocab.get("<PAD>", -1)
    # Get target padding index for cross_entropy loss, default to -100 if not PAD
    loss_ignore_index = -100  # Standard ignore index for padding in loss

    pbar_eval = tqdm(data_loader, desc=description, leave=False)

    # Check if the assumed output layer exists
    if not hasattr(primary_model, "output_layer"):
        error_msg = (
            "Primary model does not have an 'output_layer' attribute. "
            "This function assumes the hook captures activations right before it. "
            "Please adapt the function for your model structure."
        )
        logger.error(error_msg)
        raise AttributeError(error_msg)

    for batch_data in pbar_eval:
        # Adapt based on how data_loader yields batches
        # Assuming it yields: abilities, weapons, targets, _
        if len(batch_data) == 4:
            abilities, weapons, targets, _ = batch_data
        else:
            logger.error(
                f"Unexpected batch format from data loader in {description}. Expected 4 elements, got {len(batch_data)}."
            )
            continue  # Skip batch

        abilities = abilities.to(device)
        weapons = weapons.to(device)
        targets = targets.to(device)  # Targets are needed now
        key_padding_mask = (abilities == pad_id).to(device)

        batch_size = abilities.size(0)
        if batch_size == 0:
            continue

        # 1) Get Original Activations (x) via Hook
        hook.clear_activations()
        try:
            _ = primary_model(
                abilities, weapons, key_padding_mask=key_padding_mask
            )
            x = hook.get_and_clear()
        except Exception as e:
            logger.error(
                f"Error during primary model forward pass in {description}: {e}",
                exc_info=True,
            )
            continue  # Skip batch on error

        # Validate activations
        if (
            x is None
            or x.dim() != 2
            or x.shape[0] != batch_size
            or x.shape[1] != sae_config.input_dim
        ):
            logger.warning(
                f"Skipping batch in {description} due to invalid original activations shape: {x.shape if x is not None else 'None'} for batch size {batch_size}"
            )
            continue

        # 2) Get Reconstructed Activations (x_hat) from SAE
        try:
            x_hat, _ = sae_model(x.float())  # Get reconstruction
        except Exception as e:
            logger.error(
                f"Error during SAE forward pass in {description}: {e}",
                exc_info=True,
            )
            continue  # Skip batch on error

        # 3) Get Logits using both x and x_hat
        try:
            original_logits = primary_model.output_layer(x.float())
            reconstructed_logits = primary_model.output_layer(
                x_hat.float()
            )  # Use reconstructed acts
        except Exception as e:
            logger.error(
                f"Error during output layer forward pass in {description}: {e}",
                exc_info=True,
            )
            continue  # Skip batch on error

        # Ensure logits have the expected shape [batch_size, vocab_size]
        if (
            original_logits.dim() != 2
            or original_logits.shape[0] != batch_size
            or reconstructed_logits.dim() != 2
            or reconstructed_logits.shape[0] != batch_size
        ):
            logger.warning(
                f"Skipping batch in {description} due to unexpected logit shapes. Original: {original_logits.shape}, Recon: {reconstructed_logits.shape}"
            )
            continue

        # 4) Calculate Comparison Metrics

        # --- Logit Comparison ---
        batch_logit_mse = F.mse_loss(
            original_logits, reconstructed_logits
        ).item()
        batch_logit_cosine_sim = (
            F.cosine_similarity(original_logits, reconstructed_logits, dim=1)
            .mean()
            .item()
        )
        log_p_recon = F.log_softmax(reconstructed_logits, dim=-1)
        p_orig = F.softmax(original_logits, dim=-1)
        batch_kl_div = F.kl_div(
            log_p_recon, p_orig, reduction="batchmean", log_target=False
        ).item()  # log_target=False is default but explicit

        # --- Task Performance Comparison ---
        # **FIX:** Adapt target processing for potentially multi-dimensional targets
        # Assume we compare against the first target token if targets are 2D [batch, seq_len]
        targets_for_comparison = targets
        if targets.dim() == 2 and targets.shape[1] > 0:
            logger.debug(
                f"Targets have shape {targets.shape}. Using targets[:, 0] for loss/accuracy."
            )
            targets_for_comparison = targets[:, 0]  # Select the first token
        elif targets.dim() != 1:
            logger.warning(
                f"Targets have unexpected shape {targets.shape}. Skipping loss/accuracy calculation for this batch."
            )
            targets_for_comparison = None  # Flag to skip loss/acc

        batch_orig_loss = np.nan
        batch_recon_loss = np.nan
        batch_top1_acc_orig = np.nan
        batch_top1_acc_recon = np.nan
        batch_top1_match = np.nan
        current_batch_valid_samples = 0

        if (
            targets_for_comparison is not None
            and targets_for_comparison.dim() == 1
        ):
            # Prepare 1D targets for loss calculation
            targets_for_loss = targets_for_comparison.clone()
            targets_for_loss[targets_for_loss == pad_id] = loss_ignore_index

            # Calculate Cross-Entropy Loss
            try:
                batch_orig_loss = F.cross_entropy(
                    original_logits,
                    targets_for_loss,
                    ignore_index=loss_ignore_index,
                ).item()
                batch_recon_loss = F.cross_entropy(
                    reconstructed_logits,
                    targets_for_loss,
                    ignore_index=loss_ignore_index,
                ).item()
            except Exception as loss_err:
                logger.warning(
                    f"Error calculating CE loss in {description}: {loss_err}. Logits: {original_logits.shape}, Targets: {targets_for_loss.shape}"
                )
                batch_orig_loss = np.nan
                batch_recon_loss = np.nan

            # Calculate Accuracy and Prediction Match
            orig_preds = torch.argmax(original_logits, dim=-1)
            recon_preds = torch.argmax(reconstructed_logits, dim=-1)

            # Create mask for valid targets (1D)
            valid_target_mask = targets_for_loss != loss_ignore_index
            current_batch_valid_samples = valid_target_mask.sum().item()

            if current_batch_valid_samples > 0:
                # **FIX:** Index 1D tensors with 1D boolean mask - this should now work
                batch_top1_acc_orig = (
                    (
                        orig_preds[valid_target_mask]
                        == targets_for_loss[valid_target_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
                batch_top1_acc_recon = (
                    (
                        recon_preds[valid_target_mask]
                        == targets_for_loss[valid_target_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
                batch_top1_match = (
                    (
                        orig_preds[valid_target_mask]
                        == recon_preds[valid_target_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
            else:
                # Handle case where all targets in the batch are padding/ignored
                batch_top1_acc_orig = np.nan
                batch_top1_acc_recon = np.nan
                batch_top1_match = np.nan

        # 5) Accumulate Metrics
        # Accumulate per-sample metrics (weighted by batch size)
        total_logit_mse += batch_logit_mse * batch_size
        total_logit_cosine_sim += batch_logit_cosine_sim * batch_size
        total_kl_div += batch_kl_div * batch_size

        # Accumulate loss/accuracy only for valid samples
        if current_batch_valid_samples > 0:
            if not np.isnan(batch_orig_loss):
                total_orig_loss += batch_orig_loss * current_batch_valid_samples
            if not np.isnan(batch_recon_loss):
                total_recon_loss += (
                    batch_recon_loss * current_batch_valid_samples
                )
            if not np.isnan(batch_top1_acc_orig):
                total_top1_acc_orig += (
                    batch_top1_acc_orig * current_batch_valid_samples
                )
            if not np.isnan(batch_top1_acc_recon):
                total_top1_acc_recon += (
                    batch_top1_acc_recon * current_batch_valid_samples
                )
            if not np.isnan(batch_top1_match):
                total_top1_match += (
                    batch_top1_match * current_batch_valid_samples
                )
            valid_samples_for_acc_loss += current_batch_valid_samples

        num_batches += 1
        num_samples += batch_size  # Total sequences processed

        pbar_eval.set_postfix(
            logit_mse=batch_logit_mse,
            logit_cos_sim=batch_logit_cosine_sim,
            kl_div=batch_kl_div,
            recon_loss=batch_recon_loss,  # Show recon loss per batch
        )

    if num_samples == 0:
        logger.warning(
            f"{description} completed without processing any samples."
        )
        return {}  # Return empty dict if no samples processed

    # Calculate average metrics
    avg_metrics = {
        # Averaged over all samples
        "logit_mse": (
            total_logit_mse / num_samples if num_samples > 0 else np.nan
        ),
        "logit_cosine_similarity": (
            total_logit_cosine_sim / num_samples if num_samples > 0 else np.nan
        ),
        "logit_kl_divergence": (
            total_kl_div / num_samples if num_samples > 0 else np.nan
        ),  # D_KL(P_recon || P_orig)
        # Averaged only over samples with valid targets
        "original_ce_loss": (
            total_orig_loss / valid_samples_for_acc_loss
            if valid_samples_for_acc_loss > 0
            else np.nan
        ),
        "reconstructed_ce_loss": (
            total_recon_loss / valid_samples_for_acc_loss
            if valid_samples_for_acc_loss > 0
            else np.nan
        ),
        "original_top1_accuracy": (
            total_top1_acc_orig / valid_samples_for_acc_loss
            if valid_samples_for_acc_loss > 0
            else np.nan
        ),
        "reconstructed_top1_accuracy": (
            total_top1_acc_recon / valid_samples_for_acc_loss
            if valid_samples_for_acc_loss > 0
            else np.nan
        ),
        "top1_prediction_match_rate": (
            total_top1_match / valid_samples_for_acc_loss
            if valid_samples_for_acc_loss > 0
            else np.nan
        ),
        "num_eval_batches": num_batches,
        "num_eval_samples": num_samples,
        "num_valid_loss_acc_samples": valid_samples_for_acc_loss,
    }

    # Calculate loss increase if possible
    orig_loss = avg_metrics["original_ce_loss"]
    recon_loss = avg_metrics["reconstructed_ce_loss"]
    if not np.isnan(orig_loss) and not np.isnan(recon_loss):
        avg_metrics["ce_loss_increase"] = recon_loss - orig_loss
        avg_metrics["ce_loss_increase_percent"] = (
            (avg_metrics["ce_loss_increase"] / abs(orig_loss)) * 100
            if abs(orig_loss) > 1e-9
            else 0.0
        )
    else:
        avg_metrics["ce_loss_increase"] = np.nan
        avg_metrics["ce_loss_increase_percent"] = np.nan

    return avg_metrics
