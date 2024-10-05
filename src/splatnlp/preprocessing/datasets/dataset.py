import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MaskedSetDataset(Dataset):
    def __init__(
        self,
        sets: list[list[int]],
        vocab_size: int,
        mask_token_id: int,
        num_masks_per_set: int = 5,
        skew_factor: float = 1.2,
    ):
        """
        Args:
            sets: List of lists, where each sublist is a set of token IDs.
            vocab_size: Total number of tokens in your vocabulary.
            mask_token_id: The ID used to represent the MASK token.
            num_masks_per_set: Number of masked instances to generate per set.
            skew_factor: Adjust this value to control the skew of the
                mask distribution. Higher values increase the probability of
                generating more masks.
        """
        self.sets = sets
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_masks_per_set = num_masks_per_set
        self.skew_factor = skew_factor

    def __len__(self):
        return len(self.sets) * self.num_masks_per_set

    def weighted_random_masks(self, max_masks):
        """
        Generate a random number of masks with a slight preference for higher values.
        """
        # Create a triangular distribution
        x = np.arange(1, max_masks + 1)
        distribution = x / x.sum()

        distribution = distribution**self.skew_factor
        distribution /= distribution.sum()

        return np.random.choice(x, p=distribution)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        set_idx: int = idx // self.num_masks_per_set
        original_set = self.sets[set_idx].copy()
        set_length = len(original_set)
        max_masks = max(1, set_length - 1)

        masked_set = original_set.copy()
        num_masks = int(self.weighted_random_masks(max_masks))
        mask_indices = random.sample(range(set_length), num_masks)

        # Apply masks
        for mask_idx in mask_indices:
            masked_set[mask_idx] = self.mask_token_id

        # Convert to tensors
        input_tensor = torch.tensor(masked_set, dtype=torch.long)
        target_tensor = torch.tensor(original_set, dtype=torch.long)

        return input_tensor, target_tensor


def create_collate_fn(PAD_ID: int):
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        input_lengths = [len(seq) for seq in inputs]
        max_length = max(input_lengths)

        # Pad sequences
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=PAD_ID
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=PAD_ID
        )

        # Create attention masks
        attention_masks = torch.zeros(len(inputs), max_length, dtype=torch.bool)
        for i, length in enumerate(input_lengths):
            attention_masks[i, :length] = True

        return padded_inputs, padded_targets, attention_masks

    return collate_fn
