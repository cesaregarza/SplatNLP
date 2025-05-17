import random
import numpy as np
import pandas as pd
import torch

from splatnlp.preprocessing.datasets.dataset import SetDataset, create_collate_fn


def make_df():
    return pd.DataFrame(
        {
            "ability_tags": [[1, 2, 3], [4, 5]],
            "weapon_id": [10, 20],
        }
    )


def test_dataset_len_and_num_instances():
    ds = SetDataset(make_df(), vocab_size=50, num_instances_per_set=2)
    assert ds.num_instances_per_set == 2
    assert len(ds) == 2 * 2


def test_dataset_getitem_and_collate():
    random.seed(0)
    np.random.seed(0)
    ds = SetDataset(make_df(), vocab_size=50, num_instances_per_set=1)
    inp, weapon, target = ds[0]
    assert set(inp.tolist()).issubset(set(target.tolist()))
    assert weapon.item() == 10
    collate = create_collate_fn(PAD_ID=0)
    batch = collate([ds[0], ds[1]])
    inputs, weapons, targets, masks = batch
    assert inputs.shape[0] == 2
    assert weapons.tolist() == [[10], [20]]
    assert masks.dtype == torch.bool
