from dataclasses import dataclass
from typing import NamedTuple
from collections import deque

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from splatnlp.model.models import SetCompletionModel


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder following
    'Towards Monospecificity' paper.
    """

    # Basic architecture
    input_dim: int
    expansion_factor: float = 8.0  # Paper uses wider hidden layer

    # Loss coefficient
    l1_coefficient: float = 1e-5  # L1 regularization on activations

    # Dead neuron handling
    dead_neuron_threshold: float = 1e-6
    resample_interval: int = 12500  # Paper checks every 12.5k steps

    # Optimization
    learning_rate: float = 1e-4  # Paper recommends lower learning rates


class SAEOutput(NamedTuple):
    """Structured output from SAE forward pass."""

    reconstructed: torch.Tensor
    hidden: torch.Tensor
    pre_activation: torch.Tensor


class ActivationHook:
    def __init__(self):
        self.activation_inputs: torch.Tensor | None = None

    def __call__(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activation_inputs = input[0].detach()

    def get_and_clear(self) -> torch.Tensor | None:
        activation = self.activation_inputs
        self.activation_inputs = None
        return activation


def register_activation_hook(
    model: SetCompletionModel, hook: ActivationHook
) -> None:
    """Register the activation hook on the last feedforward layer."""
    last_set_transformer_layer = model.transformer_layers[-1]
    last_set_transformer_layer.feedforward[-2].register_forward_hook(hook)


class ActivationBuffer:
    def __init__(self, max_size=819200):  # As specified in the paper
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.current_size = 0
        self.broadcast_size = False

    def add(self, activations: torch.Tensor):
        """
        Add activations to buffer, handling sequence data by flattening appropriately.

        Args:
            activations: Tensor of shape (batch_size, seq_len, hidden_dim)
                        or (batch_size, hidden_dim)
        """
        # Handle different input shapes
        if len(activations.shape) == 3:
            # Shape: (batch_size, seq_len, hidden_dim)
            # Reshape to (batch_size * seq_len, hidden_dim)
            activations = activations.reshape(-1, activations.shape[-1])
        elif len(activations.shape) == 2:
            # Shape: (batch_size, hidden_dim)
            pass
        else:
            raise ValueError(
                f"Unexpected activation shape: {activations.shape}"
            )

        # Convert to CPU and detach
        activations = activations.detach().cpu()

        # Add each activation vector independently
        for activation in activations:
            if self.current_size < self.max_size:
                self.buffer.append(activation)
                self.current_size += 1
            else:
                if not self.broadcast_size:
                    print(
                        f"Buffer full, replacing elements after {self.max_size}"
                    )
                    self.broadcast_size = True
                # Replace random element
                idx = torch.randint(0, self.max_size, (1,)).item()
                self.buffer[idx] = activation

    def get_loader(self, batch_size=1024):
        if not self.buffer:
            return None

        # Convert buffer to tensor
        activations = torch.stack(list(self.buffer))
        dataset = TensorDataset(activations)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return self.current_size
