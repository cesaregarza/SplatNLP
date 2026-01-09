from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from splatnlp.model.models import SetCompletionModel
from splatnlp.model_embeddings.extract import extract_embeddings_from_dataloader
from splatnlp.preprocessing.datasets.dataset import SetDataset, create_collate_fn


def build_embedding_dataloader(
    df: pd.DataFrame,
    *,
    vocab_size: int,
    pad_token_id: int,
    num_instances_per_set: int = 5,
    skew_factor: float = 1.2,
    null_token_id: int | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = SetDataset(
        df=df,
        vocab_size=vocab_size,
        num_instances_per_set=num_instances_per_set,
        skew_factor=skew_factor,
        null_token=null_token_id,
    )
    collate_fn = create_collate_fn(PAD_ID=pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def extract_training_embeddings(
    model: SetCompletionModel,
    dataloader: DataLoader,
    *,
    pad_token_id: int,
    output_dir: str,
    device: str = "cpu",
    normalize: bool = True,
) -> dict[str, Path]:
    return extract_embeddings_from_dataloader(
        model,
        dataloader,
        pad_token_id=pad_token_id,
        output_dir=output_dir,
        device=device,
        normalize=normalize,
    )
