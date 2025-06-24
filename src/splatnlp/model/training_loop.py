import os
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from splatnlp.model.config import TrainingConfig
from splatnlp.model.utils import (
    create_multi_hot_targets,
    update_epoch_metrics,
    update_progress_bar,
)
from splatnlp.utils.constants import PAD


class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_model(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
    scaler: GradScaler | None = None,
    metric_update_interval: int = 1,
    ddp: bool = False,
) -> tuple[dict[str, dict[str, list[float]]], torch.nn.Module]:
    device = torch.device(config.device)
    model.to(device)
    if ddp and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
        )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    early_stopping = EarlyStopping(patience=config.patience)

    metrics_history = {
        "train": {
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "hamming": [],
        },
        "val": {
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "hamming": [],
        },
    }

    best_model = None
    best_val_f1 = 0

    start_time = time.time()
    for epoch in range(config.num_epochs):
        # Re‑seed DistributedSampler for true shuffling each epoch
        if config.distributed and torch.distributed.is_initialized():
            train_dl.sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        if verbose:
            print(f"Epoch {epoch + 1}/{config.num_epochs}")

        train_metrics = train_epoch(
            model,
            train_dl,
            optimizer,
            criterion,
            config,
            vocab,
            pad_token,
            verbose,
            scaler,
            metric_update_interval,
            ddp,
        )
        val_metrics = validate(
            model,
            val_dl,
            criterion,
            config,
            vocab,
            pad_token,
            verbose,
            metric_update_interval,
            ddp,
        )

        update_metrics_history(metrics_history, "train", train_metrics)
        update_metrics_history(metrics_history, "val", val_metrics)

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        estimated_time_left = avg_epoch_time * (config.num_epochs - epoch - 1)

        if verbose:
            print(
                f"Train Loss: {train_metrics['loss']:.3f}, Train F1: {train_metrics['f1']:.3f}"
            )
            print(
                f"Val Loss: {val_metrics['loss']:.3f}, Val F1: {val_metrics['f1']:.3f}"
            )
            print(f"Epoch time: {epoch_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            minutes, seconds = divmod(estimated_time_left, 60)
            hours, minutes = divmod(minutes, 60)
            print(
                f"Estimated time left: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
            )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model = model.state_dict()

        scheduler.step(val_metrics["loss"])
        if early_stopping(val_metrics["loss"]):
            if verbose:
                print("Early stopping triggered")
            if ddp and torch.distributed.is_initialized():
                torch.distributed.barrier()
            break

    if (not ddp) or (torch.distributed.get_rank() == 0):
        model.load_state_dict(best_model)
    return metrics_history, model


def train_epoch(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
    scaler: GradScaler = None,
    metric_update_interval: int = 1,
    ddp: bool = False,
) -> dict[str, float]:
    model.train()
    device = torch.device(config.device)
    epoch_metrics = {
        "loss": 0,
        "f1": 0,
        "precision": 0,
        "recall": 0,
        "hamming": 0,
    }
    all_targets, all_preds = [], []

    if verbose:
        train_iter = tqdm(train_dl, desc="Training")
    else:
        train_iter = train_dl

    for i, (batch_inputs, batch_weapons, batch_targets, _) in enumerate(
        train_iter
    ):
        batch_inputs, batch_weapons, batch_targets = (
            batch_inputs.to(device, non_blocking=True),
            batch_weapons.to(device, non_blocking=True),
            batch_targets.to(device, non_blocking=True),
        )
        key_padding_mask = (batch_inputs == vocab[pad_token]).to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        if scaler:
            with autocast(device_type=config.device, dtype=torch.float8_e4m3):
                outputs = model(
                    batch_inputs,
                    batch_weapons,
                    key_padding_mask=key_padding_mask,
                )
                target_multi_hot = create_multi_hot_targets(
                    batch_targets, vocab, device
                )
                loss = criterion(outputs, target_multi_hot)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.clip_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                batch_inputs, batch_weapons, key_padding_mask=key_padding_mask
            )
            target_multi_hot = create_multi_hot_targets(
                batch_targets, vocab, device
            )
            loss = criterion(outputs, target_multi_hot)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.clip_grad_norm
            )
            optimizer.step()

        epoch_metrics["loss"] += loss.item()

        if ((i + 1) % metric_update_interval == 0) or (i == len(train_dl) - 1):
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_targets.append(target_multi_hot.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            if verbose:
                update_progress_bar(train_iter, epoch_metrics, i + 1)

    epoch_metrics["loss"] /= len(train_dl)
    update_epoch_metrics(
        epoch_metrics, np.vstack(all_targets), np.vstack(all_preds)
    )

    # Metric reduction for distributed training
    if ddp and torch.distributed.is_initialized():
        tensor = torch.tensor(
            [
                epoch_metrics[m]
                for m in ("loss", "f1", "precision", "recall", "hamming")
            ],
            device=device,
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= torch.distributed.get_world_size()
        (
            epoch_metrics["loss"],
            epoch_metrics["f1"],
            epoch_metrics["precision"],
            epoch_metrics["recall"],
            epoch_metrics["hamming"],
        ) = tensor.tolist()

    return epoch_metrics


def validate(
    model: torch.nn.Module,
    val_dl: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
    metric_update_interval: int = 1,
    ddp: bool = False,
) -> dict[str, float]:
    model.eval()
    device = torch.device(config.device)
    epoch_metrics = {
        "loss": 0,
        "f1": 0,
        "precision": 0,
        "recall": 0,
        "hamming": 0,
    }
    all_targets, all_preds = [], []

    with torch.no_grad():
        if verbose:
            val_iter = tqdm(val_dl, desc="Validation")
        else:
            val_iter = val_dl

        for i, (batch_inputs, batch_weapons, batch_targets, _) in enumerate(
            val_iter
        ):
            batch_inputs, batch_weapons, batch_targets = (
                batch_inputs.to(device, non_blocking=True),
                batch_weapons.to(device, non_blocking=True),
                batch_targets.to(device, non_blocking=True),
            )
            key_padding_mask = (batch_inputs == vocab[pad_token]).to(
                device, non_blocking=True
            )

            outputs = model(
                batch_inputs, batch_weapons, key_padding_mask=key_padding_mask
            )
            target_multi_hot = create_multi_hot_targets(
                batch_targets, vocab, device
            )

            loss = criterion(outputs, target_multi_hot)
            epoch_metrics["loss"] += loss.item()

            if ((i + 1) % metric_update_interval == 0) or (
                i == len(val_dl) - 1
            ):
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                all_targets.append(target_multi_hot.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

                if verbose:
                    update_progress_bar(val_iter, epoch_metrics, i + 1)

    epoch_metrics["loss"] /= len(val_dl)
    update_epoch_metrics(
        epoch_metrics, np.vstack(all_targets), np.vstack(all_preds)
    )

    # Metric reduction for distributed training
    if ddp and torch.distributed.is_initialized():
        tensor = torch.tensor(
            [
                epoch_metrics[m]
                for m in ("loss", "f1", "precision", "recall", "hamming")
            ],
            device=device,
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= torch.distributed.get_world_size()
        (
            epoch_metrics["loss"],
            epoch_metrics["f1"],
            epoch_metrics["precision"],
            epoch_metrics["recall"],
            epoch_metrics["hamming"],
        ) = tensor.tolist()

    return epoch_metrics


def update_metrics_history(
    metrics_history: dict[str, dict[str, list[float]]],
    split: str,
    epoch_metrics: dict[str, float],
) -> None:
    for metric, value in epoch_metrics.items():
        metrics_history[split][metric].append(value)
