import logging
import time

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def inference(
    model: nn.Module,
    target: list[str],
    weapon_id: str,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    weapon_vocab: dict[str, int],
    pad_token_id: int,
) -> tuple[list[tuple[str, float]], float]:
    start_time = time.time()
    logging.info("Starting inference")
    logging.info(f"Target: {target}")
    logging.info(f"Weapon ID: {weapon_id}")
    model.eval()
    input_tokens = torch.tensor(
        [vocab[token] for token in target], device="cpu"
    ).unsqueeze(0)
    input_weapons = torch.tensor(
        [weapon_vocab[weapon_id]], device="cpu"
    ).unsqueeze(0)
    key_padding_mask = (input_tokens == pad_token_id).to("cpu")

    with torch.no_grad():
        outputs = model(
            input_tokens, input_weapons, key_padding_mask=key_padding_mask
        )
        preds = torch.sigmoid(outputs).squeeze().numpy()

    if preds.ndim == 0:
        preds = np.array([preds])

    out = [(inv_vocab[i], float(pred)) for i, pred in enumerate(preds)]
    time_taken = time.time() - start_time
    logging.info(f"Finished inference in {time_taken:.2f}s")
    return out, time_taken


def normalized_entropy(preds: np.ndarray) -> float:
    probs = preds / np.sum(preds)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(preds))
    return entropy / max_entropy
