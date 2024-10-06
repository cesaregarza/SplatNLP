import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab_size: int,
        num_instances_per_set: int = 5,
        skew_factor: float = 1.2,
    ):
        """
        Args:
            df: DataFrame with 'ability_tags' and 'weapon_id' columns.
            vocab_size: Total number of tokens in your vocabulary.
            num_instances_per_set: Number of instances to generate per set.
            skew_factor: Adjust this value to control the skew of the
                removal distribution. Higher values increase the probability of
                generating more removals.
        """
        self.ability_tags = df["ability_tags"].tolist()
        self.weapon_ids = df["weapon_id"].tolist()
        self.vocab_size = vocab_size
        self.num_instances_per_set = num_instances_per_set
        self.skew_factor = skew_factor

    def __len__(self):
        return len(self.ability_tags) * self.num_instances_per_set

    def weighted_random_removals(self, max_removals):
        """Generate a random number of removals with a slight preference for
        higher values.
        """
        # Create a triangular distribution
        x = np.arange(1, max_removals + 1)
        distribution = x / x.sum()

        distribution = distribution**self.skew_factor
        distribution /= distribution.sum()

        return np.random.choice(x, p=distribution)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        set_idx: int = idx // self.num_instances_per_set
        original_set = self.ability_tags[set_idx].copy()
        weapon_id = self.weapon_ids[set_idx]
        set_length = len(original_set)
        max_removals = max(1, set_length - 1)

        num_removals = int(self.weighted_random_removals(max_removals))
        removal_indices = random.sample(range(set_length), num_removals)

        # Remove tokens
        modified_set = [
            token
            for i, token in enumerate(original_set)
            if i not in removal_indices
        ]

        # Convert to tensors
        input_tensor = torch.tensor(modified_set, dtype=torch.long)
        target_tensor = torch.tensor(original_set, dtype=torch.long)
        weapon_tensor = torch.tensor([weapon_id], dtype=torch.long)

        return input_tensor, weapon_tensor, target_tensor


def create_collate_fn(PAD_ID: int):
    def collate_fn(batch):
        inputs, weapons, targets = zip(*batch)
        input_lengths = [len(seq) for seq in inputs]
        max_length = max(input_lengths)

        # Pad sequences
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=PAD_ID
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=PAD_ID
        )

        # Stack weapon tensors
        weapon_tensor = torch.stack(weapons)

        # Create attention masks
        attention_masks = torch.zeros(len(inputs), max_length, dtype=torch.bool)
        for i, length in enumerate(input_lengths):
            attention_masks[i, :length] = True

        return padded_inputs, weapon_tensor, padded_targets, attention_masks

    return collate_fn
