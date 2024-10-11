import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from splatnlp.preprocessing.constants import PAD


def create_multi_hot_targets(
    batch_targets: torch.Tensor,
    vocab: dict[str, int],
    device: torch.device,
    pad_token: str = PAD,
) -> torch.Tensor:
    batch_size = batch_targets.size(0)
    target_multi_hot = torch.zeros(batch_size, len(vocab), device=device)
    for i in range(batch_size):
        tokens = batch_targets[i][batch_targets[i] != vocab[pad_token]]
        target_multi_hot[i, tokens] = 1.0
    return target_multi_hot


def update_epoch_metrics(
    metrics: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    metrics["f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["hamming"] = hamming_loss(y_true, y_pred)


def update_progress_bar(
    pbar: tqdm, metrics: dict[str, float], total_batches: int
) -> None:
    pbar.set_postfix(
        {
            "Loss": f"{metrics['loss'] / total_batches:.3f}",
            "F1": f"{metrics['f1']:.3f}",
            "Precision": f"{metrics['precision']:.3f}",
            "Recall": f"{metrics['recall']:.3f}",
            "Hamming": f"{metrics['hamming']:.3f}",
        }
    )
