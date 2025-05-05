"""
evaluate.py - Contains evaluation functions for the Sparse Autoencoder,
including standard SAE metrics (MSE, L1 loss, sparsity) and reconstruction
impact evaluation.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemanticity_train.data_objects import (
    ActivationHook,
    SAEConfig,
)
from splatnlp.monosemanticity_train.models import SparseAutoencoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_sae_model(
    primary_model: SetCompletionModel,
    sae_model: SparseAutoencoder,
    hook: ActivationHook,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sae_config: SAEConfig,
    vocab: dict[str, int],
    description: str = "Validation",
) -> dict[str, float]:
    """
    Evaluates the SAE model on a dataset, measuring MSE, L1 loss, and sparsity.

    Args:
        primary_model: The primary model generating activations
        sae_model: The Sparse Autoencoder model to evaluate
        hook: Activation hook for capturing activations
        data_loader: DataLoader for evaluation data
        device: Device for computations
        sae_config: Configuration for the SAE
        vocab: Vocabulary dictionary
        description: Description for the progress bar

    Returns:
        dict: Dictionary of evaluation metrics
    """
    primary_model.eval()
    sae_model.eval()

    total_mse_loss = 0.0
    total_l1_loss = 0.0
    total_sparsity = 0.0
    total_feature_magnitude = 0.0
    num_batches = 0
    num_activations = 0

    pad_id = vocab.get("<PAD>", -1)

    pbar_eval: tqdm[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = tqdm(data_loader, desc=description, leave=False)

    for abilities, weapons, targets, _ in pbar_eval:
        abilities = abilities.to(device)
        weapons = weapons.to(device)
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
                "Skipping batch in %s due to invalid activations shape: %s",
                description,
                acts.shape if acts is not None else "None",
            )
            continue
        if acts.shape[0] == 0:  # Skip empty activation batches
            continue

        current_batch_size = acts.shape[0]
        num_activations += current_batch_size

        # 2) Run SAE forward pass
        out_tuple: tuple[torch.Tensor, torch.Tensor] = sae_model(acts.float())
        recon_acts, hidden_acts = out_tuple

        # 3) Calculate losses and metrics for this batch
        batch_mse_loss = F.mse_loss(recon_acts, acts.float()).item()
        batch_l1_loss = torch.norm(hidden_acts, p=1, dim=-1).mean().item()
        batch_sparsity = (hidden_acts > 1e-6).float().mean().item()
        batch_feature_mag = torch.abs(hidden_acts).mean().item()

        # Accumulate weighted losses
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

    # Calculate average metrics
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


@torch.no_grad()
def evaluate_reconstruction_impact(
    primary_model: SetCompletionModel,
    sae_model: SparseAutoencoder,
    hook: ActivationHook,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sae_config: SAEConfig,
    vocab: dict[str, int],
    description: str = "Reconstruction Impact Eval",
) -> dict[str, float]:
    """Evaluates how the reconstruction from the SAE impacts the primary model's
    logits.

    Compares original logits with reconstructed logits and checks differences in
    cross-entropy loss and top-1 accuracy if targets are available.

    Args:
        primary_model: The primary model generating activations
        sae_model: The Sparse Autoencoder model to evaluate
        hook: Activation hook for capturing activations
        data_loader: DataLoader for evaluation data
        device: Device for computations
        sae_config: Configuration for the SAE
        vocab: Vocabulary dictionary
        description: Description for the progress bar

    Returns:
        dict: Dictionary of evaluation metrics
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
    total_top1_match = 0.0
    valid_samples_for_acc_loss = 0

    num_batches = 0
    num_samples = 0

    pad_id = vocab.get("<PAD>", -1)
    loss_ignore_index = -100

    pbar_eval: tqdm[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = tqdm(data_loader, desc=description, leave=False)

    if not hasattr(primary_model, "output_layer"):
        error_msg = (
            "Primary model does not have an 'output_layer' attribute. "
            "This function assumes the hook captures activations right before "
            "it. Please adapt the function for your model structure."
        )
        logger.error(error_msg)
        raise AttributeError(error_msg)

    for batch_data in pbar_eval:
        if len(batch_data) == 4:
            abilities, weapons, targets, _ = batch_data
        else:
            logger.error(
                "Unexpected batch format from data loader in %s. Expected 4 "
                "elements, got %d.",
                description,
                len(batch_data),
            )
            continue

        abilities = abilities.to(device)
        weapons = weapons.to(device)
        targets = targets.to(device)
        key_padding_mask = abilities.eq(pad_id).to(device)

        batch_size = abilities.size(0)
        if batch_size == 0:
            continue

        # 1) Get Original Activations
        hook.clear_activations()
        try:
            _ = primary_model(
                abilities, weapons, key_padding_mask=key_padding_mask
            )
            x = hook.get_and_clear()
        except Exception as e:
            logger.error(
                "Error during primary model forward pass in %s: %s",
                description,
                e,
                exc_info=True,
            )
            continue

        if (
            x is None
            or x.dim() != 2
            or x.shape[0] != batch_size
            or x.shape[1] != sae_config.input_dim
        ):
            logger.warning(
                "Skipping batch in %s due to invalid original activations "
                "shape: %s for batch size %d",
                description,
                x.shape if x is not None else "None",
                batch_size,
            )
            continue

        # 2) Get Reconstructed Activations
        try:
            x_hat: torch.Tensor = sae_model(x.float())[0]
        except Exception as e:
            logger.error(
                "Error during SAE forward pass in %s: %s",
                description,
                e,
                exc_info=True,
            )
            continue

        # 3) Get Logits
        try:
            original_logits: torch.Tensor = primary_model.output_layer(
                x.float()
            )
            reconstructed_logits: torch.Tensor = primary_model.output_layer(
                x_hat.float()
            )
        except Exception as e:
            logger.error(
                "Error during output layer forward pass in %s: %s",
                description,
                e,
                exc_info=True,
            )
            continue

        if (
            original_logits.dim() != 2
            or original_logits.shape[0] != batch_size
            or reconstructed_logits.dim() != 2
            or reconstructed_logits.shape[0] != batch_size
        ):
            logger.warning(
                "Skipping batch in %s due to unexpected logit shapes. "
                "Original: %s, Recon: %s",
                description,
                original_logits.shape,
                reconstructed_logits.shape,
            )
            continue

        # 4) Calculate Comparison Metrics
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
        ).item()

        # Process targets
        targets_for_comparison = targets
        if targets.dim() == 2 and targets.shape[1] > 0:
            logger.debug(
                "Targets have shape %s. Using targets[:, 0] for loss/accuracy.",
                targets.shape,
            )
            targets_for_comparison = targets[:, 0]
        elif targets.dim() != 1:
            logger.warning(
                "Targets have unexpected shape %s. Skipping loss/accuracy "
                "calculation for this batch.",
                targets.shape,
            )
            targets_for_comparison = None

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
            targets_for_loss = targets_for_comparison.clone()
            targets_for_loss[targets_for_loss == pad_id] = loss_ignore_index

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
                    "Error calculating CE loss in %s: %s. Logits: %s, "
                    "Targets: %s",
                    description,
                    loss_err,
                    original_logits.shape,
                    targets_for_loss.shape,
                )
                batch_orig_loss = np.nan
                batch_recon_loss = np.nan

            orig_preds = torch.argmax(original_logits, dim=-1)
            recon_preds = torch.argmax(reconstructed_logits, dim=-1)

            valid_target_mask = targets_for_loss != loss_ignore_index
            current_batch_valid_samples = valid_target_mask.sum().item()

            if current_batch_valid_samples > 0:
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

        # 5) Accumulate Metrics
        total_logit_mse += batch_logit_mse * batch_size
        total_logit_cosine_sim += batch_logit_cosine_sim * batch_size
        total_kl_div += batch_kl_div * batch_size

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
        num_samples += batch_size

        pbar_eval.set_postfix(
            logit_mse=batch_logit_mse,
            logit_cos_sim=batch_logit_cosine_sim,
            kl_div=batch_kl_div,
            recon_loss=batch_recon_loss,
        )

    if num_samples == 0:
        logger.warning(
            f"{description} completed without processing any samples."
        )
        return {}

    # Calculate average metrics
    avg_metrics = {
        "logit_mse": (
            total_logit_mse / num_samples if num_samples > 0 else np.nan
        ),
        "logit_cosine_similarity": (
            total_logit_cosine_sim / num_samples if num_samples > 0 else np.nan
        ),
        "logit_kl_divergence": (
            total_kl_div / num_samples if num_samples > 0 else np.nan
        ),
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

    # Calculate loss increase
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
