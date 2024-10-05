import pandas as pd
from torch.utils.data import DataLoader

from splatnlp.preprocessing.constants import MASK, PAD
from splatnlp.preprocessing.datasets.dataset import (
    MaskedSetDataset,
    create_collate_fn,
)


def generate_tokenized_datasets(
    tokenized_abilities: pd.Series,
    frac: float = 0.1,
    random_state: int | None = None,
    validation_size: float = 0.1,
    test_size: float = 0.2,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Generate train, validation, and test datasets from tokenized ability
    tags.

    This function samples a fraction of the input tokenized dataset and splits
    it into train, validation, and test sets.

    Args:
        tokenized_abilities (pd.Series): The list of tokenized ability tags,
            where each item is a list of integer IDs representing ability tags.
        frac (float, optional): The fraction of the dataset to sample. Defaults
            to 0.1.
        random_state (int, optional): The random state for sampling. If None,
            the dataset is sampled without a random state. Defaults to None.
        validation_size (float, optional): The fraction of the sampled dataset
            to use for validation. Defaults to 0.1.
        test_size (float, optional): The fraction of the sampled dataset to use
            for testing. Defaults to 0.2.

    Returns:
        tuple[list[list[int]], list[list[int]], list[list[int]]]::
            - list[list[int]]: The training dataset.
            - list[list[int]]: The validation dataset.
            - list[list[int]]: The test dataset.
            Each dataset is a list of lists, where each inner list contains
            integer IDs representing tokenized ability tags.
    """
    # Sample the dataset
    if random_state is not None:
        sampled_abilities = tokenized_abilities.sample(
            frac=frac, random_state=random_state
        )
    else:
        sampled_abilities = tokenized_abilities.sample(frac=frac)

    # Split the dataset
    total_size = len(sampled_abilities)
    train_size = int(total_size * (1 - validation_size - test_size))
    validation_size = int(total_size * validation_size)
    train_X = sampled_abilities.iloc[:train_size].tolist()
    validation_X = sampled_abilities.iloc[
        train_size : train_size + validation_size
    ].tolist()
    test_X = sampled_abilities.iloc[train_size + validation_size :].tolist()
    return train_X, validation_X, test_X


def generate_dataloaders(
    train_set: list[list[int]],
    validation_set: list[list[int]],
    test_set: list[list[int]],
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    num_masks_per_set: int = 5,
    skew_factor: float = 1.2,
    **kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate DataLoaders for train, validation, and test sets.

    Args:
        train_set (list[list[int]]): The training dataset.
        validation_set (list[list[int]]): The validation dataset.
        test_set (list[list[int]]): The test dataset.
        vocab_size (int): The size of the vocabulary.
        mask_token_id (int): The ID of the mask token.
        pad_token_id (int): The ID of the padding token.
        num_masks_per_set (int, optional): Number of masked instances to
            generate per set. Defaults to 5.
        skew_factor (float, optional): Factor to control the skew of the mask
            distribution. Defaults to 1.2.
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
                MaskedSetDataset(
                    sets=dataset,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    num_masks_per_set=num_masks_per_set,
                    skew_factor=skew_factor,
                ),
                collate_fn=collate_fn,
                **dataloader_kwargs,
            )
        )

    return tuple(dataloaders)
