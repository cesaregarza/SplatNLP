import pandas as pd
from torch.utils.data import DataLoader

from splatnlp.preprocessing.constants import PAD
from splatnlp.preprocessing.datasets.dataset import (
    SetDataset,
    create_collate_fn,
)


def generate_tokenized_datasets(
    df: pd.DataFrame,
    frac: float = 0.1,
    random_state: int | None = None,
    validation_size: float = 0.1,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train, validation, and test datasets from a DataFrame containing
    tokenized ability tags and weapon IDs.

    This function samples a fraction of the input dataset and splits it into
    train, validation, and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing 'ability_tags' and
            'weapon_id' columns.
        frac (float, optional): The fraction of the dataset to sample. Defaults
            to 0.1.
        random_state (int, optional): The random state for sampling. If None,
            the dataset is sampled without a random state. Defaults to None.
        validation_size (float, optional): The fraction of the sampled dataset
            to use for validation. Defaults to 0.1.
        test_size (float, optional): The fraction of the sampled dataset to use
            for testing. Defaults to 0.2.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: The training dataset.
            - pd.DataFrame: The validation dataset.
            - pd.DataFrame: The test dataset.
            Each dataset is a DataFrame containing 'ability_tags' and
            'weapon_id' columns.
    """
    # Sample the dataset
    if random_state is not None:
        sampled_df = df.sample(frac=frac, random_state=random_state)
    else:
        sampled_df = df.sample(frac=frac)

    # Split the dataset
    total_size = len(sampled_df)
    train_size = int(total_size * (1 - validation_size - test_size))
    validation_size = int(total_size * validation_size)

    train_df = sampled_df.iloc[:train_size]
    validation_df = sampled_df.iloc[train_size : train_size + validation_size]
    test_df = sampled_df.iloc[train_size + validation_size :]

    return train_df, validation_df, test_df


def generate_dataloaders(
    train_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    test_set: pd.DataFrame,
    vocab_size: int,
    pad_token_id: int,
    num_instances_per_set: int = 5,
    skew_factor: float = 1.2,
    **kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate DataLoaders for train, validation, and test sets.

    Args:
        train_set (pd.DataFrame): The training dataset.
        validation_set (pd.DataFrame): The validation dataset.
        test_set (pd.DataFrame): The test dataset.
        vocab_size (int): The size of the vocabulary.
        pad_token_id (int): The ID of the padding token.
        num_instances_per_set (int, optional): Number of instances to
            generate per set. Defaults to 5.
        skew_factor (float, optional): Factor to control the skew of the
            removal distribution. Defaults to 1.2.
        **kwargs: Additional keyword arguments for DataLoader.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            - DataLoader: The DataLoader for the training set.
            - DataLoader: The DataLoader for the validation set.
            - DataLoader: The DataLoader for the test set.
    """
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_SHUFFLE = True
    DEFAULT_NUM_WORKERS = 0
    DEFAULT_DROP_LAST = True

    collate_fn = create_collate_fn(PAD_ID=pad_token_id)
    dataloader_kwargs = {
        "batch_size": kwargs.get("batch_size", DEFAULT_BATCH_SIZE),
        "shuffle": kwargs.get("shuffle", DEFAULT_SHUFFLE),
        "num_workers": kwargs.get("num_workers", DEFAULT_NUM_WORKERS),
        "drop_last": kwargs.get("drop_last", DEFAULT_DROP_LAST),
        **kwargs,
    }

    dataloaders = []
    for dataset in [train_set, validation_set, test_set]:
        dataloaders.append(
            DataLoader(
                SetDataset(
                    df=dataset,
                    vocab_size=vocab_size,
                    num_instances_per_set=num_instances_per_set,
                    skew_factor=skew_factor,
                ),
                collate_fn=collate_fn,
                **dataloader_kwargs,
            )
        )

    return tuple(dataloaders)
