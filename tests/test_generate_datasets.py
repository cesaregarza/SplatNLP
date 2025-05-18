import pandas as pd
import torch
from torch.utils.data import DataLoader

from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)


def test_generate_tokenized_datasets_and_dataloaders():
    df = pd.DataFrame(
        {
            "ability_tags": [
                [1],
                [2, 3],
                [4, 5, 6],
                [7, 8],
                [9, 10, 11],
                [12],
                [13, 14],
                [15, 16, 17, 18],
                [19],
                [20, 21],
            ],
            "weapon_id": list(range(10)),
        }
    )

    train_df, val_df, test_df = generate_tokenized_datasets(
        df, frac=1.0, random_state=0, validation_size=0.1, test_size=0.2
    )

    assert len(train_df) == 7
    assert len(val_df) == 1
    assert len(test_df) == 2

    loaders = generate_dataloaders(
        train_df,
        val_df,
        test_df,
        vocab_size=50,
        pad_token_id=99,
        batch_size=2,
        shuffle=False,
    )

    assert isinstance(loaders, tuple)
    assert len(loaders) == 3

    def pad_used(collate_fn: callable) -> bool:
        sample = [
            (torch.tensor([1]), torch.tensor([0]), torch.tensor([1])),
            (torch.tensor([2, 3]), torch.tensor([0]), torch.tensor([2, 3])),
        ]
        inputs, _, _, _ = collate_fn(sample)
        return (inputs == 99).any().item()

    for loader in loaders:
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2
        assert pad_used(loader.collate_fn)
