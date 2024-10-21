import time
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from splatnlp.model.config import TrainingConfig
from splatnlp.model.training_loop import EarlyStopping, update_metrics_history
from splatnlp.model.utils import create_multi_hot_targets
from splatnlp.preprocessing.constants import PAD
from splatnlp.research.models import ModifiedSetCompletionModel


def freeze_all_except_autoencoder(model: ModifiedSetCompletionModel):
    for name, param in model.named_parameters():
        if "sparse_autoencoder" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def train_autoencoder(
    model: ModifiedSetCompletionModel,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
    scaler: Optional[GradScaler] = None,
) -> tuple[dict[str, dict[str, list[float]]], ModifiedSetCompletionModel]:
    device = torch.device(config.device)
    model.to(device)

    # Freeze all parameters except the autoencoder
    freeze_all_except_autoencoder(model)

    # Only optimize the autoencoder parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.MSELoss()
    task_criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    early_stopping = EarlyStopping(patience=config.patience)

    metrics_history = {
        "train": {"loss": [], "task_loss": []},
        "val": {"loss": [], "task_loss": []},
    }

    best_model = None
    best_val_loss = float("inf")

    start_time = time.time()
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        if verbose:
            print(f"Epoch {epoch + 1}/{config.num_epochs}")

        train_loss, train_task_loss = train_autoencoder_epoch(
            model,
            train_dl,
            optimizer,
            criterion,
            task_criterion,
            config,
            vocab,
            pad_token,
            verbose,
            scaler,
        )
        val_loss, val_task_loss = validate_autoencoder(
            model,
            val_dl,
            criterion,
            task_criterion,
            config,
            vocab,
            pad_token,
            verbose,
        )

        update_metrics_history(
            metrics_history,
            "train",
            {"loss": train_loss, "task_loss": train_task_loss},
        )
        update_metrics_history(
            metrics_history,
            "val",
            {"loss": val_loss, "task_loss": val_task_loss},
        )

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        estimated_time_left = avg_epoch_time * (config.num_epochs - epoch - 1)

        if verbose:
            print(
                f"Train Autoencoder Loss: {train_loss:.6f}, Train Task Loss: {train_task_loss:.6f}"
            )
            print(
                f"Val Autoencoder Loss: {val_loss:.6f}, Val Task Loss: {val_task_loss:.6f}"
            )
            print(f"Epoch time: {epoch_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            minutes, seconds = divmod(estimated_time_left, 60)
            hours, minutes = divmod(minutes, 60)
            print(
                f"Estimated time left: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        scheduler.step(val_loss)
        if early_stopping(val_loss):
            if verbose:
                print("Early stopping triggered")
            break

    model.load_state_dict(best_model)
    return metrics_history, model


def train_autoencoder_epoch(
    model: ModifiedSetCompletionModel,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    task_criterion: nn.Module,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
    scaler: Optional[GradScaler] = None,
) -> tuple[float, float]:
    model.eval()
    device = torch.device(config.device)
    total_loss = 0
    total_task_loss = 0

    if verbose:
        train_iter = tqdm(train_dl, desc="Training")
    else:
        train_iter = train_dl

    for batch_inputs, batch_weapons, batch_targets, _ in train_iter:
        batch_inputs, batch_weapons, batch_targets = (
            batch_inputs.to(device),
            batch_weapons.to(device),
            batch_targets.to(device),
        )
        key_padding_mask = (batch_inputs == vocab[pad_token]).to(device)

        optimizer.zero_grad()

        if scaler:
            with autocast():
                # Get outputs from pretrained and modified models
                with torch.no_grad():
                    base_outputs = model.pretrained_model(
                        batch_inputs,
                        batch_weapons,
                        key_padding_mask=key_padding_mask,
                    )
                outputs = model(
                    batch_inputs,
                    batch_weapons,
                    key_padding_mask=key_padding_mask,
                )

                # Compute autoencoder loss
                loss = criterion(outputs, base_outputs.detach())

                # Compute task loss (for monitoring purposes only)
                target_multi_hot = create_multi_hot_targets(
                    batch_targets, vocab, device
                )
                task_loss = task_criterion(outputs, target_multi_hot)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.clip_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            # Get outputs from pretrained and modified models
            with torch.no_grad():
                base_outputs = model.pretrained_model(
                    batch_inputs,
                    batch_weapons,
                    key_padding_mask=key_padding_mask,
                )
            outputs = model(
                batch_inputs, batch_weapons, key_padding_mask=key_padding_mask
            )

            # Compute autoencoder loss
            loss = criterion(outputs, base_outputs.detach())

            # Compute task loss (for monitoring purposes only)
            target_multi_hot = create_multi_hot_targets(
                batch_targets, vocab, device
            )
            task_loss = task_criterion(outputs, target_multi_hot)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.clip_grad_norm
            )
            optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()

        if verbose:
            train_iter.set_postfix(
                {
                    "AE Loss": f"{loss.item():.6f}",
                    "Task Loss": f"{task_loss.item():.6f}",
                }
            )

    return total_loss / len(train_dl), total_task_loss / len(train_dl)


def validate_autoencoder(
    model: ModifiedSetCompletionModel,
    val_dl: torch.utils.data.DataLoader,
    criterion: nn.Module,
    task_criterion: nn.Module,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
) -> tuple[float, float]:
    model.eval()
    device = torch.device(config.device)
    total_loss = 0
    total_task_loss = 0

    with torch.no_grad():
        if verbose:
            val_iter = tqdm(val_dl, desc="Validation")
        else:
            val_iter = val_dl

        for batch_inputs, batch_weapons, batch_targets, _ in val_iter:
            batch_inputs, batch_weapons, batch_targets = (
                batch_inputs.to(device),
                batch_weapons.to(device),
                batch_targets.to(device),
            )
            key_padding_mask = (batch_inputs == vocab[pad_token]).to(device)

            # Get outputs from pretrained and modified models
            base_outputs = model.pretrained_model(
                batch_inputs, batch_weapons, key_padding_mask=key_padding_mask
            )
            outputs = model(
                batch_inputs, batch_weapons, key_padding_mask=key_padding_mask
            )

            # Compute autoencoder loss
            loss = criterion(outputs, base_outputs)

            # Compute task loss (for monitoring purposes only)
            target_multi_hot = create_multi_hot_targets(
                batch_targets, vocab, device
            )
            task_loss = task_criterion(outputs, target_multi_hot)

            total_loss += loss.item()
            total_task_loss += task_loss.item()

            if verbose:
                val_iter.set_postfix(
                    {
                        "AE Loss": f"{loss.item():.6f}",
                        "Task Loss": f"{task_loss.item():.6f}",
                    }
                )

    return total_loss / len(val_dl), total_task_loss / len(val_dl)


# Usage example remains the same as in the previous version
