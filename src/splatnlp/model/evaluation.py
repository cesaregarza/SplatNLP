import numpy as np
import torch
from tqdm import tqdm

from splatnlp.model.config import TrainingConfig
from splatnlp.model.utils import (
    create_multi_hot_targets,
    update_epoch_metrics,
    update_progress_bar,
)
from splatnlp.utils.constants import PAD


def test_model(
    model: torch.nn.Module,
    test_dl: torch.utils.data.DataLoader,
    config: TrainingConfig,
    vocab: dict[str, int],
    pad_token: str = PAD,
    verbose: bool = True,
) -> dict[str, float]:
    model.eval()
    device = torch.device(config.device)
    test_metrics = {
        "loss": 0,
        "f1": 0,
        "precision": 0,
        "recall": 0,
        "hamming": 0,
    }
    all_targets, all_preds = [], []
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        if verbose:
            test_iter = tqdm(test_dl, desc="Testing")
        else:
            test_iter = test_dl
        for i, (batch_inputs, batch_weapons, batch_targets, _) in enumerate(
            test_iter
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
            test_metrics["loss"] += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()

            all_targets.append(target_multi_hot.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            if verbose:
                update_progress_bar(test_iter, test_metrics, i + 1)

    test_metrics["loss"] /= len(test_dl)
    update_epoch_metrics(
        test_metrics, np.vstack(all_targets), np.vstack(all_preds)
    )
    return test_metrics
