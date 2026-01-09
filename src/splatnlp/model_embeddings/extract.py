from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import orjson
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.data_objects import ActivationHook


def _coerce_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value) if hasattr(value, "__iter__") else [value]


def normalize_ability_tags(
    ability_tags: Iterable | str,
    vocab: dict[str, int],
    pad_id: int,
    null_id: int | None,
) -> list[int]:
    if isinstance(ability_tags, str):
        tags = orjson.loads(ability_tags)
    else:
        tags = _coerce_list(ability_tags)
    if not tags:
        if null_id is None:
            raise ValueError(
                "Empty ability_tags require <NULL> in the vocabulary"
            )
        return [null_id]

    normalized: list[int] = []
    for tag in tags:
        if isinstance(tag, str):
            if tag.isdigit():
                tag_id = int(tag)
            else:
                tag_id = vocab.get(tag)
                if tag_id is None:
                    raise ValueError(f"Unknown ability token: {tag}")
        else:
            tag_id = int(tag)

        if tag_id == pad_id:
            raise ValueError("ability_tags must not include <PAD> tokens")
        if null_id is not None and tag_id == null_id:
            if len(tags) != 1:
                raise ValueError(
                    "<NULL> is only allowed as a single-token empty build"
                )
        normalized.append(tag_id)
    return normalized


def normalize_weapon_id(
    weapon_id: str | int,
    weapon_vocab: dict[str, int],
    weapon_token_ids: set[int],
) -> int:
    if isinstance(weapon_id, str):
        if weapon_id in weapon_vocab:
            return weapon_vocab[weapon_id]
        if weapon_id.isdigit():
            weapon_id = int(weapon_id)
        else:
            raise ValueError(f"Unknown weapon identifier: {weapon_id}")

    weapon_id = int(weapon_id)
    if weapon_id in weapon_token_ids:
        return weapon_id

    token_key = f"weapon_id_{weapon_id}"
    if token_key in weapon_vocab:
        return weapon_vocab[token_key]
    raise ValueError(f"Unknown weapon identifier: {weapon_id}")


def build_batch_tensors(
    ability_lists: list[list[int]],
    weapon_ids: list[int],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(tokens) for tokens in ability_lists)
    batch_size = len(ability_lists)

    input_tokens = torch.full(
        (batch_size, max_len),
        pad_id,
        device=device,
        dtype=torch.long,
    )
    for row_idx, tokens in enumerate(ability_lists):
        input_tokens[row_idx, : len(tokens)] = torch.tensor(
            tokens, device=device, dtype=torch.long
        )
    key_padding_mask = input_tokens == pad_id

    input_weapons = torch.tensor(
        weapon_ids,
        device=device,
        dtype=torch.long,
    ).unsqueeze(1)
    return input_tokens, input_weapons, key_padding_mask


def extract_embeddings(
    model: SetCompletionModel,
    df: pd.DataFrame,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    output_dir: Path | str,
    *,
    batch_size: int = 512,
    device: str = "cpu",
    normalize: bool = True,
    limit: int | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    total = len(df)
    hidden_dim = model.output_layer.in_features
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"

    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype="float32",
        shape=(total, hidden_dim),
    )

    try:
        with metadata_path.open("wb") as meta_handle:
            for start in tqdm(range(0, total, batch_size), desc="Embedding"):
                end = min(start + batch_size, total)
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
                    _ = model(
                        input_tokens,
                        input_weapons,
                        key_padding_mask=key_padding_mask,
                    )
                    pooled = hook.get_and_clear()

                if pooled is None:
                    raise RuntimeError("Failed to capture pooled embeddings")

                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=-1)

                embeddings[start:end, :] = pooled.detach().cpu().numpy()

                for row_idx, (weapon_id, ability_ids) in enumerate(
                    zip(weapon_ids, ability_lists)
                ):
                    source_index = batch_df.index[row_idx]
                    try:
                        source_index = int(source_index)
                    except (TypeError, ValueError):
                        source_index = str(source_index)

                    record = {
                        "row_id": start + row_idx,
                        "source_index": source_index,
                        "weapon_id": weapon_id,
                        "ability_ids": ability_ids,
                    }
                    meta_handle.write(orjson.dumps(record) + b"\n")
    finally:
        handle.remove()
        embeddings.flush()

    return {
        "embeddings": embeddings_path,
        "metadata": metadata_path,
    }


def extract_embeddings_from_dataloader(
    model: SetCompletionModel,
    dataloader: DataLoader,
    *,
    pad_token_id: int,
    output_dir: Path | str,
    device: str = "cpu",
    normalize: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    hook = ActivationHook(capture="input")
    handle = model.output_layer.register_forward_hook(hook)

    total = len(dataloader.dataset)
    hidden_dim = model.output_layer.in_features
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"

    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype="float32",
        shape=(total, hidden_dim),
    )

    row_id = 0
    try:
        with metadata_path.open("wb") as meta_handle:
            for batch in tqdm(dataloader, desc="Embedding"):
                if len(batch) == 4:
                    inputs, weapons, targets, attention_masks = batch
                else:
                    raise ValueError(
                        "Expected batch to have 4 elements "
                        "(inputs, weapons, targets, attention_masks)"
                    )

                inputs = inputs.to(torch_device)
                weapons = weapons.to(torch_device)
                attention_masks = attention_masks.to(torch_device)
                key_padding_mask = ~attention_masks

                hook.clear_activations()
                with torch.inference_mode():
                    _ = model(
                        inputs,
                        weapons,
                        key_padding_mask=key_padding_mask,
                    )
                    pooled = hook.get_and_clear()

                if pooled is None:
                    raise RuntimeError("Failed to capture pooled embeddings")

                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=-1)

                batch_size = inputs.size(0)
                embeddings[row_id : row_id + batch_size, :] = (
                    pooled.detach().cpu().numpy()
                )

                inputs_cpu = inputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                attention_cpu = attention_masks.detach().cpu()
                weapons_cpu = weapons.detach().cpu()

                for i in range(batch_size):
                    ability_ids = inputs_cpu[i][attention_cpu[i]].tolist()
                    target_mask = targets_cpu[i] != pad_token_id
                    target_ids = targets_cpu[i][target_mask].tolist()

                    record = {
                        "row_id": row_id + i,
                        "weapon_id": int(weapons_cpu[i].item()),
                        "ability_ids": ability_ids,
                        "target_ids": target_ids,
                    }
                    meta_handle.write(orjson.dumps(record) + b"\n")

                row_id += batch_size

        if row_id != total:
            raise RuntimeError(
                f"Expected {total} rows but wrote {row_id} embeddings"
            )
    finally:
        handle.remove()
        embeddings.flush()

    return {
        "embeddings": embeddings_path,
        "metadata": metadata_path,
    }
