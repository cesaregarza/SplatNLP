from __future__ import annotations

import pandas as pd
import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.model_embeddings.extract import (
    build_batch_tensors,
    normalize_ability_tags,
    normalize_weapon_id,
)
from splatnlp.monosemantic_sae.data_objects import ActivationHook


def verify_embeddings_vs_logits(
    model: SetCompletionModel,
    df: pd.DataFrame,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    *,
    batch_size: int = 64,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, float]:
    if limit is not None:
        df = df.head(limit)

    pad_id = vocab.get("<PAD>")
    if pad_id is None:
        raise ValueError("'<PAD>' token missing from vocabulary")
    null_id = vocab.get("<NULL>")
    weapon_token_ids = set(weapon_vocab.values())

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    hook = ActivationHook(capture="input")
    handle = model.output_layer.register_forward_hook(hook)

    max_abs_error = 0.0
    sum_abs_error = 0.0
    num_elements = 0

    try:
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]

            ability_lists = [
                normalize_ability_tags(row, vocab, pad_id, null_id)
                for row in batch_df["ability_tags"]
            ]
            weapon_ids = [
                normalize_weapon_id(row, weapon_vocab, weapon_token_ids)
                for row in batch_df["weapon_id"]
            ]

            input_tokens, input_weapons, key_padding_mask = (
                build_batch_tensors(
                    ability_lists, weapon_ids, pad_id, torch_device
                )
            )

            hook.clear_activations()
            with torch.inference_mode():
                logits = model(
                    input_tokens,
                    input_weapons,
                    key_padding_mask=key_padding_mask,
                )
                pooled = hook.get_and_clear()

            if pooled is None:
                raise RuntimeError("Failed to capture pooled embeddings")

            logits_manual = model.output_layer(pooled)
            diff = (logits - logits_manual).abs()
            max_abs_error = max(max_abs_error, diff.max().item())
            sum_abs_error += diff.sum().item()
            num_elements += diff.numel()
    finally:
        handle.remove()

    mean_abs_error = sum_abs_error / max(1, num_elements)
    return {
        "max_abs_error": float(max_abs_error),
        "mean_abs_error": float(mean_abs_error),
        "num_elements": int(num_elements),
    }
