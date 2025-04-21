from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion_factor: float = 8,
        l1_coefficient: float = 1e-3,
        dead_neuron_threshold: float = 1e-6,
        dead_neuron_steps: int = 12500,
        target_usage: float = 0.05,   # fraction of samples to fire
        usage_coeff: float = 1e-3,    # penalty multiplier for usage KL
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * expansion_factor)
        self.l1_coefficient = l1_coefficient
        self.dead_neuron_threshold = dead_neuron_threshold
        self.dead_neuron_steps = dead_neuron_steps
        self.target_usage = target_usage
        self.usage_coeff = usage_coeff

        # Track usage with an EMA
        self.register_buffer("usage_ema", torch.zeros(self.hidden_dim))

        # Pre-encoder bias
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        # Core autoencoder
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_uniform_(
            self.encoder.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_uniform_(
            self.decoder.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder_bias)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, input_dim]  or possibly [batch_size * seq_len, input_dim]
        Returns: (reconstruction, hidden)
        """
        x_centered = x - self.decoder_bias  # pre-encoder bias
        h = F.relu(self.encoder(x_centered))
        # Normalize decoder weights
        normalized_decoder_weights = F.normalize(self.decoder.weight, dim=0)
        reconstruction = F.linear(h, normalized_decoder_weights) + self.decoder_bias
        return reconstruction, h

    def compute_loss(
        self, x: torch.Tensor, reconstruction: torch.Tensor, hidden: torch.Tensor
    ):
        """
        Returns (total_loss, metrics_dict)
        """
        # MSE reconstruction loss
        mse_loss = F.mse_loss(reconstruction, x, reduction="mean")

        # L1 penalty on activations (optional)
        l1_loss = hidden.abs().sum(dim=1).mean()

        # Ensure hidden is [batch_size, hidden_dim]
        # If hidden is [batch_size, seq_len, hidden_dim], flatten it:
        if hidden.dim() == 3:
            # e.g. [B, S, H] -> [B*S, H]
            hidden = hidden.reshape(-1, hidden.size(-1))

        # usage_batch => fraction of samples in which each neuron > 0
        # shape => [hidden_dim]
        usage_batch = (hidden > 0).float().mean(dim=0)

        # Check dimension match
        if usage_batch.shape[0] != self.hidden_dim:
            raise RuntimeError(
                f"usage_batch has shape {usage_batch.shape}, "
                f"expected [{self.hidden_dim}]. Check your hidden shape!"
            )

        # Update usage_ema
        alpha = 0.99
        with torch.no_grad():
            self.usage_ema = alpha * self.usage_ema + (1 - alpha) * usage_batch

        # usage_ema in [hidden_dim], clamp to avoid log(0)
        p = torch.clamp(self.usage_ema, min=1e-7, max=1.0 - 1e-7)
        rho = self.target_usage

        # KL usage
        kl_usage = rho * torch.log(rho / p) + (1 - rho) * torch.log((1 - rho) / (1 - p))
        kl_loss = kl_usage.mean()

        # Weighted sum
        total_loss = mse_loss + self.l1_coefficient * l1_loss + self.usage_coeff * kl_loss

        metrics = {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
            "l1": l1_loss.item(),
            "kl_usage": kl_loss.item(),
        }
        return total_loss, metrics

    def training_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Single training step for external loop.
        """
        reconstruction, hidden = self(x)
        loss, metrics = self.compute_loss(x, reconstruction, hidden)
        optimizer.zero_grad()
        loss.backward()

        # remove parallel gradients from decoder weight, if you're doing that
        self.remove_parallel_gradients()

        optimizer.step()

        # re-normalize decoder weights
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

        return metrics

    def remove_parallel_gradients(self):
        """Optional step to remove gradient components parallel to dictionary vectors"""
        if self.decoder.weight.grad is not None:
            normalized_dict = F.normalize(self.decoder.weight.data, dim=0)
            parallel_component = (
                self.decoder.weight.grad * normalized_dict
            ).sum(0) * normalized_dict
            self.decoder.weight.grad -= parallel_component
