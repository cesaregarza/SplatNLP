import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import splatnlp.model.evaluation as evaluation
from splatnlp.model.config import TrainingConfig
from splatnlp.model.training_loop import train_epoch, validate
from splatnlp.preprocessing.datasets.dataset import (
    SetDataset,
    create_collate_fn,
)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, weapon_vocab_size: int):
        super().__init__()
        self.ability_embed = torch.nn.Embedding(vocab_size, 4)
        self.weapon_embed = torch.nn.Embedding(weapon_vocab_size, 4)
        self.fc = torch.nn.Linear(4, vocab_size)

    def forward(self, abilities: torch.Tensor, weapons: torch.Tensor, key_padding_mask=None):
        x = self.ability_embed(abilities).sum(dim=1) + self.weapon_embed(weapons).squeeze(1)
        return self.fc(x)


def make_dataloader(vocab_size: int = 10) -> DataLoader:
    random.seed(0)
    np.random.seed(0)
    df = pd.DataFrame({"ability_tags": [[1, 2, 3], [3, 4, 5]], "weapon_id": [0, 0]})
    dataset = SetDataset(df, vocab_size=vocab_size, num_instances_per_set=1)
    collate = create_collate_fn(PAD_ID=0)
    return DataLoader(dataset, batch_size=2, collate_fn=collate)


def make_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        patience=1,
        learning_rate=0.01,
        weight_decay=0.0,
        clip_grad_norm=1.0,
        scheduler_factor=0.1,
        scheduler_patience=1,
        device="cpu",
    )


def test_train_validate_and_test_model_runs():
    vocab = {"<PAD>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    dataloader = make_dataloader(vocab_size=len(vocab))
    model = DummyModel(vocab_size=len(vocab), weapon_vocab_size=1)
    config = make_config()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_metrics = train_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        config,
        vocab,
        verbose=False,
    )
    assert set(train_metrics.keys()) == {"loss", "f1", "precision", "recall", "hamming"}

    val_metrics = validate(
        model,
        dataloader,
        criterion,
        config,
        vocab,
        verbose=False,
    )
    assert set(val_metrics.keys()) == {"loss", "f1", "precision", "recall", "hamming"}

    test_metrics = evaluation.test_model(
        model, dataloader, config, vocab, verbose=False
    )
    assert set(test_metrics.keys()) == {"loss", "f1", "precision", "recall", "hamming"}
